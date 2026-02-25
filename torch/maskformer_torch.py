from dataclasses import dataclass
import math
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


@dataclass
class MaskFormerConfig:
    backbone_config: dataclass | None = None
    num_channels: int = 3
    num_queries: int = 100

    encoder_layer: int = 6
    encoder_ffn_dim: int = 2048
    encoder_attention_heads: int = 8

    decoder_layers: int = 6
    decoder_ffn_dim: int = 2048
    decoder_attention_heads: int = 8

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

    output_attentions: bool = False
    output_hidden_states: bool = False


class DETRAttention(nn.Module):
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

    def _shape(self, tensor: torch.Tensor, seq_len: int, batch_size: int):
        return (
            tensor.view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def with_pos_embed(self, tensor: torch.Tensor, object_queries: Tensor | None):
        return tensor if object_queries is None else tensor + object_queries

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        object_queries: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        spatial_position_embeddings: torch.Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.size()
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
        query_states = self._shape(query_states, target_len, batch_size).view(
            *proj_shape
        )
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        source_len = key_states.size(1)

        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (batch_size * self.num_heads, target_len, source_len):
            raise ValueError(
                f"Attention weights should be of size "
                f"{(batch_size * self.num_heads, target_len, source_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, target_len, source_len):
                raise ValueError(
                    f"Attention mask should be of size "
                    f"{(batch_size, 1, target_len, source_len)}, but is"
                    f" {attention_mask.size()}"
                )
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.zeros_like(
                    attention_mask, dtype=attn_weights.dtype
                ).masked_fill_(attention_mask, -torch.inf)
            attn_weights = (
                attn_weights.view(batch_size, self.num_heads, target_len, source_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(
                batch_size * self.num_heads, target_len, source_len
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(
                batch_size, self.num_heads, target_len, source_len
            )
            attn_weights = attn_weights_reshaped.view(
                batch_size * self.num_heads, target_len, source_len
            )
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (
            batch_size * self.num_heads,
            target_len,
            self.head_dim,
        ):
            raise ValueError(
                f"`attn_output` should be of size "
                f"{(batch_size, self.num_heads, target_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(
            batch_size, self.num_heads, target_len, self.head_dim
        )
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(batch_size, target_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class DETRDecoderLayer(nn.Module):
    def __init__(self, config: MaskFormerConfig) -> None:
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = DETRAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.dropout = config.dropout

        # NOTE: Some kind of placeholder; use more formal expression in `lucid`
        self.activation_fn: Callable[[Tensor], Tensor] | None = getattr(
            F, config.activation_function, None
        )
        if self.activation_fn is None:
            raise ValueError(
                f"Invalid activation function: '{config.activation_function}'"
            )

        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = DETRAttention(
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
        return torch.zeros(
            (batch_size, 1, target_len, source_len),
            dtype=input_embeds.dtype,
            device=device,
        )

    if attention_mask.device != device:
        attention_mask = attention_mask.to(device)

    if attention_mask.dim() == 2:
        if attention_mask.shape != (batch_size, source_len):
            raise ValueError(
                "2D attention_mask must have shape "
                f"{(batch_size, source_len)}, got {tuple(attention_mask.shape)}."
            )
        if attention_mask.dtype == torch.bool:
            attention_mask = ~attention_mask
        else:
            attention_mask = attention_mask <= 0
        attention_mask = attention_mask[:, None, None, :].expand(
            batch_size, 1, target_len, source_len
        )
        return attention_mask

    if attention_mask.dim() == 3:
        if attention_mask.shape != (batch_size, target_len, source_len):
            raise ValueError(
                "3D attention_mask must have shape "
                f"{(batch_size, target_len, source_len)}, "
                f"got {tuple(attention_mask.shape)}."
            )
        if attention_mask.dtype == torch.bool:
            attention_mask = ~attention_mask
        else:
            attention_mask = attention_mask <= 0
        return attention_mask[:, None, :, :]

    if attention_mask.dim() == 4:
        expected = (batch_size, 1, target_len, source_len)
        if attention_mask.shape != expected:
            raise ValueError(
                f"4D attention_mask must have shape {expected}, "
                f"got {tuple(attention_mask.shape)}."
            )
        return attention_mask

    raise ValueError(
        f"Unsupported attention_mask ndim={attention_mask.dim()}. "
        "Expected 2D, 3D, or 4D."
    )


class DETRDecoder(nn.Module):
    def __init__(self, config: MaskFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop

        self.layers = nn.ModuleList(
            [DETRDecoderLayer(config) for _ in range(config.decoder_layers)]
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
                dropout_probability = torch.rand([])
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
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    prob = inputs.sigmoid()

    cross_entropy_loss_pos = criterion(inputs, torch.ones_like(inputs))
    focal_pos = ((1 - prob) ** gamma) * cross_entropy_loss_pos
    focal_pos *= alpha

    cross_entropy_loss_neg = criterion(inputs, torch.zeros_like(inputs))
    focal_neg = (prob**gamma) * cross_entropy_loss_neg
    focal_neg *= 1 - alpha

    loss = torch.matmul(focal_pos, labels.T) + torch.matmul(focal_neg, (1 - labels).T)
    return loss / height_and_width


def pairwise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    inputs = inputs.sigmoid().flatten(dim=1)
    numerator = 2 * torch.matmul(inputs, labels.T)

    denominator = inputs.sum(dim=-1)[:, None] + labels.sum(dim=-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_focal_loss(
    inputs: Tensor,
    labels: Tensor,
    num_masks: int,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    probs = inputs.sigmoid()

    cross_entropy_loss = criterion(inputs, labels)
    p_t = probs * labels + (1 - probs) * (1 - labels)
    loss = cross_entropy_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
        loss = alpha_t * loss

    loss = loss.mean(dim=1).sum() / num_masks
    return loss


def dice_loss(inputs: Tensor, labels: Tensor, num_masks: int) -> Tensor:
    probs = inputs.sigmoid().flatten(dim=1)
    numerator = 2 * (probs * labels).sum(dim=-1)
    denominator = probs.sum(dim=-1) + labels.sum(dim=-1)

    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def linear_sum_assignment(cost_matrix: Tensor) -> tuple[Tensor, Tensor]:
    if cost_matrix.dim() != 2:
        raise ValueError(
            f"cost_matrix must be 2D, got shape={tuple(cost_matrix.shape)}"
        )

    n_rows, n_cols = cost_matrix.shape
    out_device = cost_matrix.device

    if n_rows == 0 or n_cols == 0:
        empty = torch.empty(0, dtype=torch.long, device=out_device)
        return empty, empty

    size = max(n_rows, n_cols)
    cost = cost_matrix.detach().to(device="cpu", dtype=torch.float64).clone()
    padded = torch.zeros((size, size), dtype=cost.dtype, device="cpu")
    padded[:n_rows, :n_cols] = cost

    mask = torch.zeros((size, size), dtype=torch.int8, device="cpu")
    row_cover = torch.zeros(size, dtype=torch.bool, device="cpu")
    col_cover = torch.zeros(size, dtype=torch.bool, device="cpu")
    path = torch.zeros((size * 2, 2), dtype=torch.long, device="cpu")
    eps = 1e-9

    def _step1() -> None:
        padded.sub_(padded.min(dim=1, keepdim=True).values)

    def _step2() -> None:
        padded.sub_(padded.min(dim=0).values)

    def _step3() -> None:
        for i in range(size):
            if row_cover[i]:
                continue
            for j in range(size):
                if torch.abs(padded[i, j]) < eps and (not col_cover[j]):
                    mask[i, j] = 1
                    row_cover[i] = True
                    col_cover[j] = True
                    break

        row_cover.fill_(False)
        col_cover.fill_(False)

    def _step4() -> bool:
        for j in range(size):
            if torch.any(mask[:, j] == 1):
                col_cover[j] = True
        return int(col_cover.sum().item()) >= min(n_rows, n_cols)

    def _find_zero() -> tuple[int, int]:
        for i in range(size):
            if row_cover[i]:
                continue
            for j in range(size):
                if torch.abs(padded[i, j]) < eps and (not col_cover[j]):
                    return i, j
        return -1, -1

    def _find_star_in_row(row: int) -> int:
        cols = torch.where(mask[row] == 1)[0]
        return int(cols[0].item()) if cols.numel() else -1

    def _find_star_in_col(col: int) -> int:
        rows = torch.where(mask[:, col] == 1)[0]
        return int(rows[0].item()) if rows.numel() else -1

    def _find_prime_in_row(row: int) -> int:
        cols = torch.where(mask[row] == 2)[0]
        return int(cols[0].item()) if cols.numel() else -1

    def _augment_path(count: int) -> None:
        for i in range(count + 1):
            r = int(path[i, 0].item())
            c = int(path[i, 1].item())
            mask[r, c] = 0 if mask[r, c] == 1 else 1

    def _clear_covers() -> None:
        row_cover.fill_(False)
        col_cover.fill_(False)

    def _erase_primes() -> None:
        mask[mask == 2] = 0

    def _step6() -> None:
        uncovered_rows = ~row_cover
        uncovered_cols = ~col_cover
        if not uncovered_rows.any() or not uncovered_cols.any():
            return

        min_val = padded[uncovered_rows][:, uncovered_cols].min()
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
        cols = torch.where(mask[i] == 1)[0]
        if cols.numel() == 0:
            continue
        col = int(cols[0].item())
        if col < n_cols:
            row_ind.append(i)
            col_ind.append(col)

    return (
        torch.tensor(row_ind, dtype=torch.long, device=out_device),
        torch.tensor(col_ind, dtype=torch.long, device=out_device),
    )


class MaskFormerHungarianMatcher(nn.Module):
    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0
    ) -> None:
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("At least one cost must be > 0.")

        self.cost_class = cost_class
        self.cost_mask = cost_mask
        self.cost_dice = cost_dice

    @torch.no_grad()
    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: Tensor,
        class_labels: Tensor,
    ) -> list[tuple[Tensor]]:
        indices: list[tuple[Tensor, Tensor]] = []

        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits

        for pred_probs, pred_mask, target_mask, labels in zip(
            preds_probs, preds_masks, mask_labels, class_labels
        ):
            target_mask = F.interpolate(
                target_mask[:, None], size=pred_mask.shape[-2:], mode="nearest"
            )
            pred_probs = pred_probs.softmax(dim=-1)

            cost_class = -pred_probs[:, labels]
            pred_mask_flat = pred_mask.flatten(dim=1)
            target_mask_flat = target_mask[:, 0].flatten(dim=1)

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
            (i.astype(torch.int64), j.astype(torch.int64)) for i, j in indices
        ]
        return matched_indices


class MaskFormerLoss(nn.Module):
    def __init__(
        self,
        num_labels: int,
        matcher: MaskFormerHungarianMatcher,
        weight_dict: dict[str, float],
        eos_coef: float,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def _max_by_axis(self, t_list: list[list[int]]) -> list[int]:
        maxes = t_list[0]
        for sublist in t_list[1]:
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

        padded_tensors = torch.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = torch.ones((B, H, W), dtype=torch.bool, device=device)

        for tensor, padded_tensor, padding_mask in zip(
            tensors, padded_tensors, padding_masks
        ):
            padded_tensor[
                : tensor.shape[0], : tensor.shape[1], : tensor.shape[2]
            ].copy_(tensor)
            padding_mask[: tensor.shape[1], : tensor.shape[2]] = False

        return padded_tensors, padding_masks

    def loss_labels(
        self,
        class_queries_logits: Tensor,
        class_labels: list[Tensor],
        indices: tuple[Tensor, Tensor],
    ) -> dict[str, Tensor]:
        pred_logits = class_queries_logits
        batch_size, num_queries, _ = pred_logits.shape

        criterion = nn.CrossEntropyLoss(weight=self.empty_weight)
        idx = self._get_predictions_permutation_indices(indices)

        target_classes_o = torch.cat(
            [target[j] for target, (_, j) in zip(class_labels, indices)]
        )
        target_classes = torch.full(
            (batch_size, num_queries),
            self.num_labels,
            dtype=torch.int64,
            device=pred_logits.device,
        )
        target_classes[idx] = target_classes_o

        pred_logits_transposed = pred_logits.transpose(1, 2)
        loss_ce = criterion(pred_logits_transposed, target_classes)

        losses = {"loss_cross_entropy": loss_ce}
        return losses

    def loss_masks(
        self,
        masks_queries_logits: Tensor,
        mask_labels: list[Tensor],
        indices: tuple[Tensor, Tensor],
        num_masks: int,
    ) -> dict[str, Tensor]:
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_target_permutation_indices(indices)

        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = F.interpolate(
            pred_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred_masks = pred_masks[:, 0].flatten(dim=1)
        target_masks = target_masks.flatten(dim=1)

        losses = {
            "loss_mask": sigmoid_focal_loss(pred_masks, target_masks, num_masks),
            "loss_dice": dice_loss(pred_masks, target_masks, num_masks),
        }
        return losses

    def _get_predictions_permutation_indices(
        self, indices: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        batch_indices = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        predictions_indices = torch.cat([src for (src, _) in indices])
        return batch_indices, predictions_indices

    def _get_target_permutation_indices(
        self, indices: tuple[Tensor, Tensor]
    ) -> tuple[Tensor, Tensor]:
        batch_indices = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        target_indices = torch.cat([tgt for (_, tgt) in indices])
        return batch_indices, target_indices

    def get_num_masks(self, class_labels: Tensor, device: torch.Device) -> Tensor:
        num_masks = sum(len(classes) for classes in class_labels)
        num_masks = torch.as_tensor(num_masks, dtype=torch.float, device=device)

        num_masks = torch.clamp(num_masks, min=1)
        return num_masks

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


class MaskFormerFPNConvLayer(nn.Module):
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


class MaskFormerFPNLayer(nn.Module):
    def __init__(self, in_features: int, lateral_features: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(
                lateral_features, in_features, kernel_size=1, padding=0, bias=False
            ),
            nn.GroupNorm(32, in_features),
        )
        self.block = MaskFormerFPNConvLayer(in_features, in_features)

    def forward(self, down: Tensor, left: Tensor) -> Tensor:
        left = self.proj(left)
        down = F.interpolate(down, size=left.shape[-2:], mode="nearest")
        down += left
        down = self.block(down)
        return down


class MaskFormerFPNModel(nn.Module):
    def __init__(
        self, in_features: int, lateral_widths: list[int], feature_size: int = 256
    ) -> None:
        super().__init__()
        self.stem = MaskFormerFPNConvLayer(in_features, feature_size)
        self.layers = nn.Sequential(
            *[
                MaskFormerFPNLayer(feature_size, lateral_width)
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


class MaskFormerPixelDecoder(nn.Module):
    def __init__(
        self, *args, feature_size: int = 256, mask_feature_size: int = 256, **kwargs
    ) -> None:
        super().__init__()
        self.fpn = MaskFormerFPNModel(*args, feature_size=feature_size, **kwargs)
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


class MaskFormerSinePositionEmbedding(nn.Module):
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

    def forward(
        self,
        shape: torch.Size,
        device: torch.Device,
        dtype: torch.dtype,
        mask: Tensor | None = None,
    ) -> Tensor:
        NotImplemented


class PredictionBlock(nn.Module):
    NotImplemented


class MaskFormerMLPPredictionHead(nn.Module):
    NotImplemented


class MaskFormerPixelLevelModule(nn.Module):
    NotImplemented


class MaskFormerTransformerModule(nn.Module):
    NotImplemented


class MaskFormer(nn.Module):
    NotImplemented
