from dataclasses import dataclass
import copy
import math
from typing import Any, Callable, ClassVar

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid import register_model

from lucid.models.vision.resnet import resnet_18, resnet_34, resnet_50, resnet_101


__all__ = [
    "MaskFormerConfig",
    "MaskFormer",
    "maskformer_resnet_18",
    "maskformer_resnet_34",
    "maskformer_resnet_50",
    "maskformer_resnet_101",
]


@dataclass
class MaskFormerConfig:
    num_labels: int
    fpn_feature_size: int
    mask_feature_size: int

    backbone_config: dict | None = None
    num_channels: int = 3
    num_queries: int = 100

    encoder_layer: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8

    decoder_config: dict | None = None
    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8
    decoder_hidden_size: int | None = None
    decoder_num_queries: int | None = None

    encoder_layerdrop: float = 0.0
    decoder_layerdrop: float = 0.0

    is_encoder_decoder: bool = True
    activation_function: str = "relu"

    d_model: int = 256
    dropout: float = 0.1

    attention_dropout: float = 0.1
    activation_dropout: float = 0.0

    init_std: float = 0.02
    init_xavier_std: float = 1.0
    dilation: bool = False

    class_cost: float = 1.0
    mask_loss_coefficient: float = 1.0
    dice_loss_coefficient: float = 1.0
    eos_coefficient: float = 0.1
    no_object_weight: float = 0.1

    output_attentions: bool = False
    output_hidden_states: bool = False


@dataclass
class _BackboneOutput:
    feature_maps: list[Tensor]


class _ConvNormLayer(nn.Module):
    def __init__(
        self,
        convolution: nn.Module,
        normalization: nn.Module,
        activation: nn.Module | None,
    ) -> None:
        super().__init__()
        self.convolution = convolution
        self.normalization = normalization
        self.activation = activation

    def forward(self, input_: Tensor) -> Tensor:
        out = self.convolution(input_)
        out = self.normalization(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class _BackboneBlock(nn.Module):
    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.shortcut: _ConvNormLayer | None = None
        if getattr(block, "downsample", None) is not None:
            self.shortcut = _ConvNormLayer(
                block.downsample[0], block.downsample[1], None
            )

        if hasattr(block, "conv1") and hasattr(block.conv1, "conv"):
            self.layer = nn.Sequential(
                _ConvNormLayer(block.conv1.conv, block.conv1.bn, block.conv1.relu),
                _ConvNormLayer(block.conv2.conv, block.conv2.bn, block.conv2.relu),
                _ConvNormLayer(block.conv3[0], block.conv3[1], None),
            )
            self.activation = block.relu
            self.se = getattr(block, "se", None)
        else:
            self.layer = nn.Sequential(
                _ConvNormLayer(block.conv1, block.bn1, block.relu1),
                _ConvNormLayer(block.conv2, block.bn2, None),
            )
            self.activation = block.relu2
            self.se = None

    def forward(self, input_: Tensor) -> Tensor:
        identity = input_
        out = input_
        for layer in self.layer:
            out = layer(out)
        if self.se is not None:
            out = self.se(out)
        if self.shortcut is not None:
            identity = self.shortcut(input_)
        out += identity
        return self.activation(out)


class _BackboneStage(nn.Module):
    def __init__(self, stage: nn.Sequential) -> None:
        super().__init__()
        self.layers = nn.Sequential(*[_BackboneBlock(block) for block in stage])

    def forward(self, input_: Tensor) -> Tensor:
        out = input_
        for layer in self.layers:
            out = layer(out)
        return out


class _MaskFormerResNetBackbone(nn.Module):
    _CHANNELS_BY_VARIANT: ClassVar[dict[str, list[int]]] = {
        "resnet_18": [64, 128, 256, 512],
        "resnet_34": [64, 128, 256, 512],
        "resnet_50": [256, 512, 1024, 2048],
        "resnet_101": [256, 512, 1024, 2048],
    }

    def __init__(
        self,
        variant: str = "resnet_50",
        in_channels: int = 3,
        dilation: bool = False,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.pretrained = pretrained
        builders = {
            "resnet_18": resnet_18,
            "resnet_34": resnet_34,
            "resnet_50": resnet_50,
            "resnet_101": resnet_101,
        }
        weight_prefix = {
            "resnet_18": "ResNet_18_Weights",
            "resnet_34": "ResNet_34_Weights",
            "resnet_50": "ResNet_50_Weights",
            "resnet_101": "ResNet_101_Weights",
        }
        if variant not in builders:
            raise ValueError(
                f"Unsupported ResNet variant '{variant}'. "
                f"Choose one of {tuple(builders.keys())}."
            )
        _ = dilation

        builder = builders[variant]
        if pretrained:
            import lucid.weights as W

            weights = getattr(W, weight_prefix[variant]).DEFAULT
            model = builder(num_classes=1000, in_channels=3, weights=weights)
            if in_channels != 3:
                stem_conv = model.stem[0]
                model.stem[0] = nn.Conv2d(
                    in_channels,
                    stem_conv.out_channels,
                    kernel_size=stem_conv.kernel_size,
                    stride=stem_conv.stride,
                    padding=stem_conv.padding,
                    bias=stem_conv.bias is not None,
                )
                nn.init.kaiming_normal(model.stem[0].weight, mode="fan_out")
                if model.stem[0].bias is not None:
                    nn.init.constant(model.stem[0].bias, 0.0)
        else:
            model = builder(num_classes=1000, in_channels=in_channels, weights=None)

        self.variant = variant
        self.channels = self._CHANNELS_BY_VARIANT[variant]

        self.embedder = nn.Module()
        self.embedder.embedder = _ConvNormLayer(
            model.stem[0], model.stem[1], model.stem[2]
        )
        self.maxpool = model.maxpool

        self.encoder = nn.Module()
        self.encoder.stages = nn.Sequential(
            _BackboneStage(model.layer1),
            _BackboneStage(model.layer2),
            _BackboneStage(model.layer3),
            _BackboneStage(model.layer4),
        )

    @classmethod
    def from_config(
        cls,
        config: MaskFormerConfig,
        variant: str = "resnet_50",
        pretrained: bool = False,
    ) -> _MaskFormerResNetBackbone:
        return cls(
            variant=variant,
            in_channels=config.num_channels,
            dilation=config.dilation,
            pretrained=pretrained,
        )

    def forward(self, pixel_values: Tensor) -> _BackboneOutput:
        hidden_states = self.embedder.embedder(pixel_values)
        hidden_states = self.maxpool(hidden_states)

        c2 = self.encoder.stages[0](hidden_states)
        c3 = self.encoder.stages[1](c2)
        c4 = self.encoder.stages[2](c3)
        c5 = self.encoder.stages[3](c4)
        return _BackboneOutput(feature_maps=[c2, c3, c4, c5])


class _DETRAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads "
                f"(got `embed_dim`: {self.embed_dim} and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: Tensor, seq_len: int, batch_size: int):
        return tensor.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose((0, 2, 1, 3))

    def with_pos_embed(self, tensor: Tensor, object_queries: Tensor | None):
        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        object_queries: Tensor | None = None,
        key_value_states: Tensor | None = None,
        spatial_position_embeddings: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.shape
        hidden_states_original = hidden_states
        key_value_states_original = key_value_states

        if object_queries is not None:
            hidden_states = self.with_pos_embed(hidden_states, object_queries)

        if spatial_position_embeddings is not None:
            key_value_states = self.with_pos_embed(
                key_value_states, spatial_position_embeddings
            )

        query_states = self.q_proj(hidden_states) * self.scaling
        if is_cross_attention:
            key_states = self._shape(self.k_proj(key_value_states), -1, batch_size)
            value_states = self._shape(
                self.v_proj(key_value_states_original), -1, batch_size
            )
        else:
            key_states = self._shape(self.k_proj(hidden_states), -1, batch_size)
            value_states = self._shape(
                self.v_proj(hidden_states_original), -1, batch_size
            )

        proj_shape = (batch_size * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, target_len, batch_size).reshape(
            *proj_shape
        )
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        source_len = key_states.shape[1]

        attn_weights = lucid.matmul(query_states, key_states.transpose((0, 2, 1)))

        if attn_weights.shape != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size "
                    f"{(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.shape}"
                )
            if attention_mask.dtype is bool:
                attention_mask = lucid.zeros_like(
                    attention_mask, dtype=attn_weights.dtype
                ).masked_fill(attention_mask, float("-inf"))
            attn_weights = (
                attn_weights.reshape(batch_size, self.num_heads, target_len, source_len)
                + attention_mask
            )
            attn_weights = attn_weights.reshape(
                batch_size * self.num_heads, target_len, source_len
            )

        attn_weights = F.softmax(attn_weights, axis=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.reshape(
                batch_size, self.num_heads, target_len, source_len
            )
            attn_weights = attn_weights_reshaped.reshape(
                batch_size * self.num_heads, target_len, source_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = lucid.matmul(attn_probs, value_states)

        if attn_output.shape != (
            batch_size * self.num_heads,
            target_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.reshape(
            batch_size, self.num_heads, target_len, self.head_dim
        )
        attn_output = attn_output.transpose((0, 2, 1, 3))
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class _DETRDecoderLayer(nn.Module):
    def __init__(self, config: MaskFormerConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = _DETRAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        self.activation_fn: Callable[[Tensor], Tensor] | None = (
            nn.utils.get_activation_from_name(config.activation_function)
        )
        if self.activation_fn is None:
            raise ValueError(
                f"Invalid activation function: '{config.activation_function}'"
            )

        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = _DETRAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        object_queries: Tensor | None = None,
        query_position_embeddings: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[Tensor] | tuple[Tensor, ...]:
        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states,
            object_queries=query_position_embeddings,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            hidden_states, cross_attn_weights = self.encoder_attn(
                hidden_states,
                object_queries=query_position_embeddings,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                spatial_position_embeddings=object_queries,
                output_attentions=output_attentions,
            )

            hidden_states = F.dropout(
                hidden_states, p=self.dropout, training=self.training
            )
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

        residual = hidden_states

        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs


def create_bidirectional_mask(
    input_embeds: Tensor,
    attention_mask: Tensor | None,
    encoder_hidden_states: Tensor | None,
) -> Tensor | None:
    if encoder_hidden_states is None:
        if attention_mask is not None:
            raise ValueError(
                "encoder_attention_mask is provided, but encoder_hidden_states is None."
            )
        return None

    batch_size, target_len = input_embeds.shape[:2]
    source_len = encoder_hidden_states.shape[1]
    device = input_embeds.device

    if attention_mask is None:
        return lucid.zeros(
            (batch_size, 1, target_len, source_len),
            dtype=input_embeds.dtype,
            device=device,
        )

    if attention_mask.device != device:
        attention_mask = attention_mask.to(device)

    if attention_mask.ndim == 2:
        if attention_mask.shape != (batch_size, source_len):
            raise ValueError(
                "2D attention_mask must have shape "
                f"{(batch_size, source_len)}, got {tuple(attention_mask.shape)}."
            )
        if attention_mask.dtype is bool:
            attention_mask = ~attention_mask
        else:
            attention_mask = attention_mask <= 0
        attention_mask = attention_mask[:, None, None, :].expand(
            batch_size, 1, target_len, source_len
        )
        return attention_mask

    if attention_mask.ndim == 3:
        if attention_mask.shape != (batch_size, target_len, source_len):
            raise ValueError(
                "3D attention_mask must have shape "
                f"{(batch_size, target_len, source_len)}, "
                f"got {tuple(attention_mask.shape)}."
            )
        if attention_mask.dtype is bool:
            attention_mask = ~attention_mask
        else:
            attention_mask = attention_mask <= 0
        return attention_mask[:, None, :, :]

    if attention_mask.ndim == 4:
        expected = (batch_size, 1, target_len, source_len)
        if attention_mask.shape != expected:
            raise ValueError(
                f"4D attention_mask must have shape {expected}, "
                f"got {tuple(attention_mask.shape)}."
            )
        return attention_mask

    raise ValueError(
        f"Unsupported attention_mask ndim={attention_mask.ndim}. "
        "Expected 2D, 3D, or 4D."
    )


class _DETRDecoder(nn.Module):
    def __init__(self, config: MaskFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.ModuleList(
            [_DETRDecoderLayer(config) for _ in range(config.decoder_layers)]
        )
        self.layernorm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_embeds: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        object_queries: Tensor | None = None,
        query_position_embeddings: Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs,
    ) -> tuple[Tensor, ...]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if input_embeds is None:
            raise ValueError("input_embeds must not be None.")
        hidden_states = input_embeds

        encoder_attention_mask = create_bidirectional_mask(
            input_embeds=input_embeds,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and encoder_hidden_states is not None) else None
        )

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = lucid.random.rand(
                    (), device=hidden_states.device
                ).item()
                if dropout_probability < self.layerdrop:
                    continue

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None,
                object_queries=object_queries,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        hidden_states = self.layernorm(hidden_states)
        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_self_attns,
                all_cross_attentions,
            ]
            if v is not None
        )


def pairwise_sigmoid_focal_loss(
    inputs: Tensor, labels: Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> Tensor:
    if alpha < 0:
        raise ValueError("alpha must be positive")

    height_and_width = inputs.shape[1]
    prob = F.sigmoid(inputs)

    cross_entropy_loss_pos = F.binary_cross_entropy_with_logits(
        inputs,
        lucid.ones_like(inputs),
        reduction=None,
    )
    focal_pos = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    cross_entropy_loss_neg = F.binary_cross_entropy_with_logits(
        inputs,
        lucid.zeros_like(inputs),
        reduction=None,
    )
    focal_neg = (prob**gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    loss = lucid.matmul(focal_pos, labels.T) + lucid.matmul(focal_neg, (1 - labels).T)
    return loss / height_and_width


def pairwise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    inputs = F.sigmoid(inputs).flatten(start_axis=1)
    numerator = 2 * lucid.matmul(inputs, labels.T)

    denominator = inputs.sum(axis=-1)[:, None] + labels.sum(axis=-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_focal_loss(
    inputs: Tensor,
    labels: Tensor,
    num_masks: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    probs = F.sigmoid(inputs)

    cross_entropy_loss = F.binary_cross_entropy_with_logits(
        inputs, labels, reduction=None
    )
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(axis=1).sum() / num_masks
    return loss


def dice_loss(inputs: Tensor, labels: Tensor, num_masks: Tensor) -> Tensor:
    probs = F.sigmoid(inputs).flatten(start_axis=1)
    numerator = 2 * (probs * labels).sum(axis=-1)
    denominator = probs.sum(axis=-1) + labels.sum(axis=-1)

    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def linear_sum_assignment(cost_matrix: Tensor) -> tuple[Tensor, Tensor]:
    if cost_matrix.ndim != 2:
        raise ValueError(
            f"cost_matrix must be 2D, got shape={tuple(cost_matrix.shape)}"
        )

    n_rows, n_cols = cost_matrix.shape
    out_device = cost_matrix.device

    if n_rows == 0 or n_cols == 0:
        empty = lucid.empty(0, dtype=lucid.Int32, device=out_device)
        return empty, empty

    size = max(n_rows, n_cols)
    cost = cost_matrix.detach()
    if cost.device != "cpu":
        cost = cost.cpu()

    cost = Tensor(Tensor.copy_data(cost.data), device="cpu")
    padded = lucid.zeros((size, size), dtype=cost.dtype, device="cpu")
    padded[:n_rows, :n_cols] = cost

    mask = lucid.zeros((size, size), dtype=lucid.Int8, device="cpu")
    row_cover = lucid.zeros(size, dtype=bool, device="cpu")
    col_cover = lucid.zeros(size, dtype=bool, device="cpu")
    path = lucid.zeros((size * 2, 2), dtype=lucid.Int32, device="cpu")
    eps = 1e-9

    def _step1() -> None:
        padded[:] = padded - lucid.min(padded, axis=1, keepdims=True)

    def _step2() -> None:
        padded[:] = padded - lucid.min(padded, axis=0)

    def _step3() -> None:
        for i in range(size):
            if row_cover[i]:
                continue
            for j in range(size):
                if lucid.abs(padded[i, j]) < eps and (not col_cover[j]):
                    mask[i, j] = 1
                    row_cover[i] = True
                    col_cover[j] = True
                    break

        row_cover[:] = False
        col_cover[:] = False

    def _step4() -> bool:
        for j in range(size):
            if (mask[:, j] == 1).any():
                col_cover[j] = True
        return int(col_cover.sum().item()) >= min(n_rows, n_cols)

    def _find_zero() -> tuple[int, int]:
        for i in range(size):
            if row_cover[i]:
                continue
            for j in range(size):
                if lucid.abs(padded[i, j]) < eps and (not col_cover[j]):
                    return i, j
        return -1, -1

    def _find_star_in_row(row: int) -> int:
        cols = lucid.nonzero(mask[row] == 1)
        return int(cols[0].item()) if cols.size else -1

    def _find_star_in_col(col: int) -> int:
        rows = lucid.nonzero(mask[:, col] == 1)
        return int(rows[0].item()) if rows.size else -1

    def _find_prime_in_row(row: int) -> int:
        cols = lucid.nonzero(mask[row] == 2)
        return int(cols[0].item()) if cols.size else -1

    def _augment_path(count: int) -> None:
        for i in range(count + 1):
            r = int(path[i, 0].item())
            c = int(path[i, 1].item())
            mask[r, c] = 0 if mask[r, c] == 1 else 1

    def _clear_covers() -> None:
        row_cover[:] = False
        col_cover[:] = False

    def _erase_primes() -> None:
        mask[mask == 2] = 0

    def _step6() -> None:
        uncovered_rows = ~row_cover
        uncovered_cols = ~col_cover
        if not uncovered_rows.any() or not uncovered_cols.any():
            return

        min_val = lucid.min(padded[uncovered_rows][:, uncovered_cols])
        padded[row_cover] += min_val
        padded[:, ~col_cover] -= min_val

    _step1()
    _step2()
    _step3()

    max_iters = size * size * 4
    for _ in range(max_iters):
        if _step4():
            break

        row, col = _find_zero()
        if row == -1:
            _step6()
            continue

        mask[row, col] = 2
        star_col = _find_star_in_row(row)
        if star_col != -1:
            row_cover[row] = True
            col_cover[star_col] = False
            continue

        path_count = 0
        path[path_count, 0] = row
        path[path_count, 1] = col

        for _ in range(size * 2):
            star_row = _find_star_in_col(int(path[path_count, 1].item()))
            if star_row == -1:
                break

            path_count += 1
            path[path_count, 0] = star_row
            path[path_count, 1] = path[path_count - 1, 1]

            prime_col = _find_prime_in_row(star_row)
            if prime_col == -1:
                break

            path_count += 1
            path[path_count, 0] = star_row
            path[path_count, 1] = prime_col

        _augment_path(path_count)
        _clear_covers()
        _erase_primes()

    row_ind, col_ind = [], []
    for i in range(n_rows):
        cols = lucid.nonzero(mask[i] == 1)
        if cols.size == 0:
            continue
        col = int(cols[0].item())
        if col < n_cols:
            row_ind.append(i)
            col_ind.append(col)

    return (
        Tensor(row_ind, dtype=lucid.Int32, device=out_device),
        Tensor(col_ind, dtype=lucid.Int32, device=out_device),
    )


class _MaskFormerHungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0
    ) -> None:
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("At least one cost must be > 0.")

        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @lucid.no_grad()
    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
    ) -> list[tuple[Tensor, Tensor]]:
        indices: list[tuple[Tensor, Tensor]] = []

        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits

        for pred_probs, pred_mask, target_mask, labels in zip(
            preds_probs, preds_masks, mask_labels, class_labels
        ):
            target_mask = F.interpolate(
                target_mask[:, None], size=pred_mask.shape[-2:], mode="nearest"
            )
            pred_probs = F.softmax(pred_probs, axis=-1)

            cost_class = -pred_probs[:, labels]
            pred_mask_flat = pred_mask.flatten(start_axis=1)
            target_mask_flat = target_mask[:, 0].flatten(start_axis=1)

            cost_mask = pairwise_sigmoid_focal_loss(pred_mask_flat, target_mask_flat)
            cost_dice = pairwise_dice_loss(pred_mask_flat, target_mask_flat)

            cost_matrix = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            assigned_indices = linear_sum_assignment(cost_matrix)
            indices.append(assigned_indices)

        matched_indices = [
            (i.astype(lucid.Int32), j.astype(lucid.Int32)) for i, j in indices
        ]
        return matched_indices


class _MaskFormerLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        matcher: _MaskFormerHungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = lucid.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _max_by_axis(self, t_list: list[list[int]]) -> list[int]:
        maxes = t_list[0]
        for sublist in t_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)

        return maxes

    def _pad_images_to_max_in_batch(
        self, tensors: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_size = len(tensors)
        batch_shape = [batch_size] + max_size
        B, _, H, W = batch_shape

        dtype = tensors[0].dtype
        device = tensors[0].device

        padded_tensors = lucid.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = lucid.ones((B, H, W), dtype=bool, device=device)

        for i, tensor in enumerate(tensors):
            c, h, w = tensor.shape
            padded_tensors[i, :c, :h, :w] = tensor
            padding_masks[i, :h, :w] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self,
        class_queries_logits: Tensor,
        class_labels: list[Tensor],
        indices: list[tuple[Tensor, Tensor]],
    ) -> dict[str, Tensor]:
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape

        idx = self._get_predictions_permutation_indices(indices)

        target_classes = lucid.full(
            (batch_size, num_queries),
            self.num_labels,
            dtype=lucid.Int32,
            device=pred_logits.device,
        )
        target_class_parts = [
            target[j] for target, (_, j) in zip(class_labels, indices) if j.size > 0
        ]
        if target_class_parts:
            target_classes_o = lucid.concatenate(tuple(target_class_parts), axis=0)
            target_classes[idx] = target_classes_o

        pred_logits_transposed = pred_logits.transpose((0, 2, 1))
        pred_logits_flat = pred_logits_transposed.reshape(
            -1, pred_logits_transposed.shape[1]
        )
        target_classes_flat = target_classes.reshape(-1)
        loss_ce = F.cross_entropy(
            pred_logits_flat, target_classes_flat, weight=self.empty_weight
        )

        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: Tensor,
        mask_labels: list[Tensor],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: Tensor,
    ) -> dict[str, Tensor]:
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_target_permutation_indices(indices)

        if src_idx[0].size == 0:
            zero = lucid.zeros(
                (), dtype=masks_queries_logits.dtype, device=masks_queries_logits.device
            )
            return {"loss_mask": zero, "loss_dice": zero}

        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = F.interpolate(
            pred_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred_masks = pred_masks[:, 0].flatten(start_axis=1)
        target_masks = target_masks.flatten(start_axis=1)

        losses = {
            "loss_mask": sigmoid_focal_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }
        return losses

    def _get_predictions_permutation_indices(
        self, indices: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        batch_indices: list[Tensor] = []
        predictions_indices: list[Tensor] = []
        device = indices[0][0].device if len(indices) > 0 else self.empty_weight.device
        for i, (src, _) in enumerate(indices):
            if src.size == 0:
                continue
            batch_indices.append(
                lucid.full(src.shape, i, dtype=lucid.Int32, device=device)
            )
            predictions_indices.append(src.astype(lucid.Int32))

        if not predictions_indices:
            return (
                lucid.empty(0, dtype=lucid.Int32, device=device),
                lucid.empty(0, dtype=lucid.Int32, device=device),
            )
        return (
            lucid.concatenate(tuple(batch_indices), axis=0),
            lucid.concatenate(tuple(predictions_indices), axis=0),
        )

    def _get_target_permutation_indices(
        self, indices: list[tuple[Tensor, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        batch_indices: list[Tensor] = []
        target_indices: list[Tensor] = []
        device = indices[0][1].device if len(indices) > 0 else self.empty_weight.device
        for i, (_, tgt) in enumerate(indices):
            if tgt.size == 0:
                continue
            batch_indices.append(
                lucid.full(tgt.shape, i, dtype=lucid.Int32, device=device)
            )
            target_indices.append(tgt.astype(lucid.Int32))

        if not target_indices:
            return (
                lucid.empty(0, dtype=lucid.Int32, device=device),
                lucid.empty(0, dtype=lucid.Int32, device=device),
            )
        return (
            lucid.concatenate(tuple(batch_indices), axis=0),
            lucid.concatenate(tuple(target_indices), axis=0),
        )

    def get_num_masks(self, class_labels: list[Tensor], device: str) -> Tensor:
        num_masks = max(sum(len(classes) for classes in class_labels), 1)
        return Tensor(float(num_masks), dtype=lucid.Float32, device=device)

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
    ) -> dict[str, Tensor]:
        indices = self.matcher(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels
        )
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)
        losses: dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        return losses


class _MaskFormerFPNConvLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.layers = [
            nn.Conv2d(
                in_features,
                out_features,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.GroupNorm(32, out_features),
            nn.ReLU(),
        ]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input_: Tensor) -> Tensor:
        hidden_state = input_
        for layer in self.layers:
            hidden_state = layer(hidden_state)

        return hidden_state


class _MaskFormerFPNLayer(nn.Module):
    def __init__(self, in_features: int, lateral_features: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                lateral_features, in_features, kernel_size=1, padding=0, bias=False
            ),
            nn.GroupNorm(32, in_features),
        )
        self.block = _MaskFormerFPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = F.interpolate(down, size=left.shape[-2:], mode="nearest")
        down += left
        down = self.block(down)
        return down


class _MaskFormerFPNModel(nn.Module):
    def __init__(
        self, in_features: int, lateral_widths: list[int], feature_size: int = 256
    ) -> None:
        super().__init__()
        self.stem = _MaskFormerFPNConvLayer(in_features, feature_size)
        self.layers = nn.Sequential(
            *[
                _MaskFormerFPNLayer(feature_size, lateral_width)
                for lateral_width in lateral_widths[::-1]
            ]
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:
        fpn_features = []
        last_feature = features[-1]
        other_featurs = features[:-1]

        output = self.stem(last_feature)
        for layer, left in zip(self.layers, other_featurs[::-1]):
            output = layer(output, left)
            fpn_features.append(output)

        return fpn_features


class _MaskFormerPixelDecoder(nn.Module):
    def __init__(
        self, *args, feature_size: int = 256, mask_feature_size: int = 256, **kwargs
    ) -> None:
        super().__init__()
        self.fpn = _MaskFormerFPNModel(*args, feature_size=feature_size, **kwargs)
        self.mask_projection = nn.Conv2d(
            feature_size, mask_feature_size, kernel_size=3, padding=1
        )

    def forward(
        self, features: list[Tensor], output_hidden_states: bool = False
    ) -> tuple[Tensor] | tuple[Tensor, ...]:
        fpn_features = self.fpn(features)
        last_feature_projected = self.mask_projection(fpn_features[-1])
        return (
            (last_feature_projected, tuple(fpn_features))
            if output_hidden_states
            else (last_feature_projected,)
        )


class _MaskFormerSinePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed.")

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale
        self._sinusoidal_embedder = nn.SinusoidalPosEmbedding(
            seq_len=None, embed_dim=num_pos_feats
        )

    def _axis_positional_embedding(
        self, length: int, device: str, dtype: Any
    ) -> Tensor:
        dummy = lucid.zeros(
            (1, length + 1, self.num_pos_feats), device=device, dtype=dtype
        )
        axis_embed = self._sinusoidal_embedder(dummy)[:, 1:, :]
        return axis_embed[0]

    def forward(
        self,
        shape: tuple[int, ...],
        device: str,
        dtype: Any,
        mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, _, height, width = shape

        if mask is None and not self.normalize:
            x_embed = self._axis_positional_embedding(width, device, dtype)
            y_embed = self._axis_positional_embedding(height, device, dtype)

            pos_x = x_embed[None, None, :, :].expand(
                batch_size, height, width, self.num_pos_feats
            )
            pos_y = y_embed[None, :, None, :].expand(
                batch_size, height, width, self.num_pos_feats
            )
            return lucid.concatenate((pos_y, pos_x), axis=3).transpose((0, 3, 1, 2))

        if mask is None:
            mask = lucid.zeros((batch_size, height, width), device=device, dtype=bool)

        not_mask = (~mask).astype(dtype)
        y_embed = lucid.cumsum(not_mask, axis=1)
        x_embed = lucid.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = lucid.arange(self.num_pos_feats, dtype=lucid.Int64, device=device)
        dim_t = dim_t.astype(dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = lucid.stack(
            (lucid.sin(pos_x[..., 0::2]), lucid.cos(pos_x[..., 1::2])), axis=4
        ).flatten(start_axis=3)

        pos_y = lucid.stack(
            (lucid.sin(pos_y[..., 0::2]), lucid.cos(pos_y[..., 1::2])), axis=4
        ).flatten(start_axis=3)

        pos = lucid.concatenate((pos_y, pos_x), axis=3).transpose((0, 3, 1, 2))
        return pos


class _PredictionBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, activation: nn.Module) -> None:
        super().__init__()
        self.layers = [nn.Linear(in_dim, out_dim), activation]
        for i, layer in enumerate(self.layers):
            self.add_module(str(i), layer)

    def forward(self, input_: Tensor) -> Tensor:
        hidden_state = input_
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class _MaskFormerMLPPredictionHead(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3
    ) -> None:
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = _PredictionBlock(in_dim, out_dim, activation=activation)

            self.layers.append(layer)
            self.add_module(str(i), layer)

    def forward(self, input_: Tensor) -> Tensor:
        hidden_states = input_
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class _MaskFormerPixelLevelModule(nn.Module):
    def __init__(self, config: MaskFormerConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.encoder = backbone

        feature_channels = self.encoder.channels
        self.decoder = _MaskFormerPixelDecoder(
            in_features=feature_channels[-1],
            feature_size=config.fpn_feature_size,
            mask_feature_size=config.mask_feature_size,
            lateral_widths=feature_channels[:-1],
        )

    def forward(
        self, pixel_values: Tensor, output_hidden_states: bool = False
    ) -> tuple[Tensor, ...]:
        features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(features, output_hidden_states)

        last_hidden_state = decoder_output[0]
        outputs = (features[-1], last_hidden_state)

        if output_hidden_states:
            hidden_states = decoder_output[1]
            outputs = outputs + (tuple(features),) + (hidden_states,)

        return outputs


class _MaskFormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: MaskFormerConfig) -> None:
        super().__init__()
        hidden_size = config.decoder_hidden_size or config.d_model
        if hidden_size != config.d_model:
            raise ValueError(
                "decoder_hidden_size must match d_model to keep decoder dimensions consistent."
            )
        num_queries = config.decoder_num_queries or config.num_queries
        should_project = in_features != hidden_size

        self.position_embedder = _MaskFormerSinePositionEmbedding(
            num_pos_feats=hidden_size // 2, normalize=True
        )
        self.queries_embedder = nn.Embedding(num_queries, hidden_size)
        self.input_projection = (
            nn.Conv2d(in_features, hidden_size, kernel_size=1)
            if should_project
            else None
        )
        self.decoder = _DETRDecoder(config)

    def forward(
        self,
        image_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, Tensor | tuple[Tensor, ...] | None]:
        if self.input_projection is not None:
            image_features = self.input_projection(image_features)

        batch_size, num_channels, height, width = image_features.shape
        object_queries = self.position_embedder(
            image_features.shape,
            image_features.device,
            image_features.dtype,
            mask=None,
        )

        queries_embeddings = self.queries_embedder.weight.unsqueeze(axis=0)
        queries_embeddings = queries_embeddings.repeat(batch_size, axis=0)
        input_embeds = lucid.zeros_like(queries_embeddings, requires_grad=self.training)

        image_features = image_features.reshape(
            batch_size, num_channels, height * width
        ).transpose((0, 2, 1))

        object_queries = object_queries.reshape(
            batch_size, num_channels, height * width
        ).transpose((0, 2, 1))

        decoder_output = self.decoder(
            input_embeds,
            encoder_hidden_states=image_features,
            encoder_attention_mask=None,
            object_queries=object_queries,
            query_position_embeddings=queries_embeddings,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        hidden_states = decoder_output[1] if output_hidden_states else None
        attentions = None
        cross_attentions = None
        if output_attentions:
            attention_idx = 2 if output_hidden_states else 1
            attentions = decoder_output[attention_idx]
            cross_attentions = (
                decoder_output[attention_idx + 1]
                if len(decoder_output) > attention_idx + 1
                else None
            )

        return {
            "last_hidden_state": decoder_output[0],
            "hidden_states": hidden_states,
            "attentions": attentions,
            "cross_attentions": cross_attentions,
        }


class _MaskFormerModel(nn.Module):
    def __init__(
        self, config: MaskFormerConfig, backbone: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.config = config
        if backbone is None:
            raise ValueError("backbone must be provided to _MaskFormerModel.")
        self.pixel_level_module = _MaskFormerPixelLevelModule(config, backbone)
        self.transformer_module = _MaskFormerTransformerModule(
            in_features=self.pixel_level_module.encoder.channels[-1], config=config
        )
        if getattr(self.pixel_level_module.encoder, "pretrained", False):
            for module in self.pixel_level_module.encoder.modules():
                module._skip_maskformer_init = True
        self.apply(self._init_weights)

    @lucid.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "_skip_maskformer_init", False):
            return

        xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        if isinstance(module, _MaskFormerTransformerModule):
            if module.input_projection is not None:
                module.input_projection._skip_default_init = True
                nn.init.xavier_uniform(module.input_projection.weight, gain=xavier_std)
                if module.input_projection.bias is not None:
                    nn.init.constant(module.input_projection.bias, 0.0)

        elif isinstance(module, _MaskFormerFPNModel):
            module.stem.layers[0]._skip_default_init = True
            nn.init.xavier_uniform(module.stem.layers[0].weight, gain=xavier_std)

        elif isinstance(module, _MaskFormerFPNLayer):
            module.proj[0]._skip_default_init = True
            nn.init.xavier_uniform(module.proj[0].weight, gain=xavier_std)

        elif isinstance(module, _MaskFormerFPNConvLayer):
            module.layers[0]._skip_default_init = True
            nn.init.xavier_uniform(module.layers[0].weight, gain=xavier_std)

        elif isinstance(module, _MaskFormerMLPPredictionHead):
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    submodule._skip_default_init = True
                    nn.init.xavier_uniform(submodule.weight, gain=xavier_std)
                    if submodule.bias is not None:
                        nn.init.constant(submodule.bias, 0.0)

        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.constant(module.bias, 0.0)
            if module.weight is not None:
                nn.init.constant(module.weight, 1.0)

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)) and not getattr(
            module, "_skip_default_init", False
        ):
            nn.init.normal(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.constant(module.bias, 0.0)

            if getattr(module, "running_mean", None) is not None:
                nn.init.constant(module.running_mean, 0.0)
                nn.init.constant(module.running_var, 1.0)

        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0

        elif isinstance(module, _MaskFormerLoss):
            empty_weight = lucid.ones(module.num_labels + 1)
            empty_weight[-1] = module.eos_coef
            module.empty_weight.data = empty_weight.data

    def forward(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        pixel_level_module_output = self.pixel_level_module(
            pixel_values, output_hidden_states
        )
        image_features = pixel_level_module_output[0]
        pixel_embeddings = pixel_level_module_output[1]

        transformer_module_output = self.transformer_module(
            image_features,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        queries = transformer_module_output["last_hidden_state"]

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        hidden_states = None

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output[2]
            pixel_decoder_hidden_states = pixel_level_module_output[3]
            transformer_decoder_hidden_states = transformer_module_output[
                "hidden_states"
            ]

            hidden_states = (
                encoder_hidden_states
                + pixel_decoder_hidden_states
                + transformer_decoder_hidden_states
            )

        output = {
            "encoder_last_hidden_state": image_features,
            "pixel_decoder_last_hidden_state": pixel_embeddings,
            "transformer_decoder_last_hidden_state": queries,
            "encoder_hidden_states": encoder_hidden_states,
            "pixel_decoder_hidden_states": pixel_decoder_hidden_states,
            "transformer_decoder_hidden_states": transformer_decoder_hidden_states,
            "hidden_states": hidden_states,
            "attentions": transformer_module_output["attentions"],
        }
        return output


class MaskFormer(nn.Module):
    def __init__(
        self, config: MaskFormerConfig, backbone: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.model = _MaskFormerModel(config, backbone=backbone)
        hidden_size = config.decoder_hidden_size or config.d_model

        self.class_predictor = nn.Linear(hidden_size, config.num_labels + 1)
        self.mask_embedder = _MaskFormerMLPPredictionHead(
            hidden_size, hidden_size, config.mask_feature_size
        )

        self.matcher = _MaskFormerHungarianMatcher(
            cost_class=config.class_cost,
            cost_dice=config.dice_loss_coefficient,
            cost_mask=config.mask_loss_coefficient,
        )

        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_cost,
            "loss_mask": config.mask_loss_coefficient,
            "loss_dice": config.dice_loss_coefficient,
        }

        self.criterion = _MaskFormerLoss(
            config.num_labels,
            matcher=self.matcher,
            weight_dict=self.weight_dict,
            eos_coef=config.no_object_weight,
        )

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
    ) -> dict[str, Tensor]:
        loss_dict: dict[str, Tensor] = self.criterion(
            masks_queries_logits, class_queries_logits, mask_labels, class_labels
        )
        for key, weight in self.weight_dict.items():
            for loss_key in list(loss_dict.keys()):
                if key in loss_key:
                    loss_dict[loss_key] = loss_dict[loss_key] * weight

        return loss_dict

    def get_loss(self, loss_dict: dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def get_logits(self, outputs: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        pixel_embeddings = outputs.get("pixel_decoder_last_hidden_state")
        if pixel_embeddings is None:
            pixel_embeddings = outputs["pixel_deocder_last_hidden_state"]
        transformer_decoder_hidden_states = outputs[
            "transformer_decoder_last_hidden_state"
        ]

        class_queries_logits = self.class_predictor(transformer_decoder_hidden_states)
        mask_embeddings = self.mask_embedder(transformer_decoder_hidden_states)

        mask_queries_logits = lucid.einops.einsum(
            "bqc,bchw->bqhw", mask_embeddings, pixel_embeddings
        )
        return class_queries_logits, mask_queries_logits

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: list[Tensor] | None = None,
        class_labels: list[Tensor] | None = None,
        pixel_mask: Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        raw_outputs = self.model(
            pixel_values,
            pixel_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )
        loss, loss_dict = None, None
        class_queries_logits, masks_queries_logits = self.get_logits(raw_outputs)

        if mask_labels is not None and class_labels is not None:
            loss_dict: dict[str, Tensor] = self.get_loss_dict(
                masks_queries_logits, class_queries_logits, mask_labels, class_labels
            )
            loss = self.get_loss(loss_dict)

        return {
            "loss": loss,
            "class_queries_logits": class_queries_logits,
            "masks_queries_logits": masks_queries_logits,
            **raw_outputs,
        }

    @lucid.no_grad()
    def predict(
        self,
        pixel_values: Tensor,
        pixel_mask: Tensor | None = None,
        output_size: tuple[int, int] | None = None,
        top_k_queries: int | None = None,
        return_logits: bool = False,
        return_scores: bool = False,
    ) -> Tensor | dict[str, Tensor]:
        was_training = self.training
        self.eval()

        outputs = self.forward(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=False,
            output_attentions=False,
        )
        class_queries_logits = outputs["class_queries_logits"]
        masks_queries_logits = outputs["masks_queries_logits"]

        all_class_probs = F.softmax(class_queries_logits, axis=-1)
        class_probs = all_class_probs[..., :-1]
        no_object_probs = all_class_probs[..., -1]
        mask_probs = F.sigmoid(masks_queries_logits)

        if top_k_queries is not None:
            num_queries = class_probs.shape[1]
            k = max(1, min(int(top_k_queries), num_queries))

            objectness = 1.0 - no_object_probs
            _, keep_idx = lucid.topk(objectness, k=k, axis=1)
            keep_mask = lucid.zeros_like(objectness, dtype=bool)

            for b in range(class_probs.shape[0]):
                keep_mask[b, keep_idx[b].astype(lucid.Int32)] = True
            class_probs = class_probs * keep_mask[:, :, None].astype(class_probs.dtype)
            mask_probs = mask_probs * keep_mask[:, :, None, None].astype(
                mask_probs.dtype
            )

        segmentation_logits = lucid.einops.einsum(
            "bqc,bqhw->bchw", class_probs, mask_probs
        )

        if output_size is None:
            output_size = (pixel_values.shape[-2], pixel_values.shape[-1])

        if segmentation_logits.shape[-2:] != output_size:
            segmentation_logits = F.interpolate(
                segmentation_logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        segmentation = lucid.argmax(segmentation_logits, axis=1).astype(lucid.Int32)

        if was_training:
            self.train()
        if not return_logits and not return_scores:
            return segmentation

        pred: dict[str, Tensor] = {"segmentation": segmentation}
        if return_logits:
            pred["logits"] = segmentation_logits
        if return_scores:
            pred["scores"] = lucid.max(F.softmax(segmentation_logits, axis=1), axis=1)
        return pred


def _maskformer_preset_config(variant: str, num_labels: int) -> MaskFormerConfig:
    base_depths = {
        "resnet_18": [2, 2, 2, 2],
        "resnet_34": [3, 4, 6, 3],
        "resnet_50": [3, 4, 6, 3],
        "resnet_101": [3, 4, 23, 3],
    }
    base_hidden_sizes = {
        "resnet_18": [64, 128, 256, 512],
        "resnet_34": [64, 128, 256, 512],
        "resnet_50": [256, 512, 1024, 2048],
        "resnet_101": [256, 512, 1024, 2048],
    }
    if variant not in base_depths:
        raise ValueError(
            f"Unsupported variant '{variant}'. "
            f"Choose one of {tuple(base_depths.keys())}."
        )

    d_model = 256
    decoder_layers = 6
    attention_heads = 8
    ff_dim = 2048
    dropout = 0.1
    attention_dropout = 0.0
    activation_dropout = 0.0
    init_std = 0.02
    init_xavier_std = 1.0

    decoder_config = {
        "model_type": "detr",
        "d_model": d_model,
        "encoder_layers": decoder_layers,
        "decoder_layers": decoder_layers,
        "encoder_attention_heads": attention_heads,
        "decoder_attention_heads": attention_heads,
        "encoder_ffn_dim": ff_dim,
        "decoder_ffn_dim": ff_dim,
        "dropout": dropout,
        "attention_dropout": attention_dropout,
        "activation_dropout": activation_dropout,
        "init_std": init_std,
        "init_xavier_std": init_xavier_std,
    }
    backbone_config = {
        "model_type": "resnet",
        "depths": base_depths[variant],
        "hidden_sizes": base_hidden_sizes[variant],
    }

    return MaskFormerConfig(
        num_labels=num_labels,
        fpn_feature_size=256,
        mask_feature_size=256,
        backbone_config=backbone_config,
        num_queries=100,
        encoder_layer=decoder_layers,
        encoder_ffn_dim=ff_dim,
        encoder_attention_heads=attention_heads,
        decoder_config=decoder_config,
        decoder_layers=decoder_layers,
        decoder_ffn_dim=ff_dim,
        decoder_attention_heads=attention_heads,
        decoder_hidden_size=d_model,
        decoder_num_queries=100,
        d_model=d_model,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        init_std=init_std,
        init_xavier_std=init_xavier_std,
        dilation=False,
        class_cost=1.0,
        mask_loss_coefficient=1.0,
        dice_loss_coefficient=1.0,
        eos_coefficient=0.1,
        no_object_weight=0.1,
    )


def _maskformer_config_to_kwargs(config: MaskFormerConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for field_name in MaskFormerConfig.__dataclass_fields__.keys():
        kwargs[field_name] = copy.deepcopy(getattr(config, field_name))
    return kwargs


def _build_maskformer_with_resnet_backbone(
    variant: str,
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> MaskFormer:
    config = _maskformer_preset_config(variant=variant, num_labels=num_labels)
    merged_kwargs = _maskformer_config_to_kwargs(config)
    merged_kwargs.update(config_kwargs)
    merged_kwargs["num_labels"] = num_labels
    config = MaskFormerConfig(**merged_kwargs)

    backbone = _MaskFormerResNetBackbone.from_config(
        config=config, variant=variant, pretrained=pretrained_backbone
    )
    return MaskFormer(config=config, backbone=backbone)


@register_model
def maskformer_resnet_18(
    num_labels: int, *, pretrained_backbone: bool = False, **config_kwargs: Any
) -> MaskFormer:
    return _build_maskformer_with_resnet_backbone(
        "resnet_18",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def maskformer_resnet_34(
    num_labels: int, *, pretrained_backbone: bool = False, **config_kwargs: Any
) -> MaskFormer:
    return _build_maskformer_with_resnet_backbone(
        "resnet_34",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def maskformer_resnet_50(
    num_labels: int, *, pretrained_backbone: bool = False, **config_kwargs: Any
) -> MaskFormer:
    return _build_maskformer_with_resnet_backbone(
        "resnet_50",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def maskformer_resnet_101(
    num_labels: int, *, pretrained_backbone: bool = False, **config_kwargs: Any
) -> MaskFormer:
    return _build_maskformer_with_resnet_backbone(
        "resnet_101",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )
