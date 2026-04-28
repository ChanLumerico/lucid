from dataclasses import dataclass, field
import copy
import math
from typing import Any, Callable, Final

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid import register_model

from lucid_legacy.models.base import PreTrainedModelMixin
from lucid_legacy.models.vision.maskformer import (
    _MaskFormerResNetBackbone,
    _maskformer_infer_resnet_variant,
    linear_sum_assignment,
)
from lucid_legacy.models.vision.swin import swin_base, swin_large, swin_small, swin_tiny

__all__ = [
    "Mask2FormerConfig",
    "Mask2Former",
    "mask2former_resnet_18",
    "mask2former_resnet_34",
    "mask2former_resnet_50",
    "mask2former_resnet_101",
    "mask2former_swin_tiny",
    "mask2former_swin_small",
    "mask2former_swin_base",
    "mask2former_swin_large",
]


_MASK_ATTENTION_FILL_VALUE: Final[float] = -1e12


@dataclass
class Mask2FormerConfig:
    num_labels: int

    feature_size: int = 256
    mask_feature_size: int = 256
    hidden_dim: int = 256

    backbone_config: dict | None = None
    num_channels: int = 3
    num_queries: int = 100

    encoder_layers: int = 6
    encoder_feedforward_dim: int = 1024

    decoder_layers: int = 10
    dim_feedforward: int = 2048
    num_attention_heads: int = 8

    feature_strides: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    common_stride: int = 4
    enforce_input_projection: bool = False

    activation_function: str = "relu"
    pre_norm: bool = False
    dropout: float = 0.0

    init_std: float = 0.02
    init_xavier_std: float = 1.0
    dilation: bool = False

    class_weight: float = 2.0
    mask_weight: float = 5.0
    dice_weight: float = 5.0
    no_object_weight: float = 0.1

    train_num_points: int = 12544
    oversample_ratio: float = 3.0
    importance_sample_ratio: float = 0.75

    use_auxiliary_loss: bool = True
    output_auxiliary_logits: bool | None = None

    output_attentions: bool = False
    output_hidden_states: bool = False


@dataclass
class _BackboneOutput:
    feature_maps: list[Tensor]


def _mask2former_infer_swin_variant(backbone_config: dict | None) -> str | None:
    if not isinstance(backbone_config, dict):
        return None
    if backbone_config.get("model_type", "") != "swin":
        return None

    embed_dim = backbone_config.get("embed_dim")
    depths = backbone_config.get("depths")
    if not isinstance(depths, list):
        return None

    if embed_dim == 96 and depths == [2, 2, 6, 2]:
        return "swin_tiny"
    if embed_dim == 96 and depths == [2, 2, 18, 2]:
        return "swin_small"
    if embed_dim == 128 and depths == [2, 2, 18, 2]:
        return "swin_base"
    if embed_dim == 192 and depths == [2, 2, 18, 2]:
        return "swin_large"
    return None


class _Mask2FormerSwinBackbone(nn.Module):
    _CHANNELS_BY_VARIANT: dict[str, list[int]] = {
        "swin_tiny": [96, 192, 384, 768],
        "swin_small": [96, 192, 384, 768],
        "swin_base": [128, 256, 512, 1024],
        "swin_large": [192, 384, 768, 1536],
    }

    def __init__(
        self,
        variant: str = "swin_small",
        in_channels: int = 3,
        image_size: int = 224,
        drop_path_rate: float = 0.3,
        window_size: int = 7,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        builders = {
            "swin_tiny": swin_tiny,
            "swin_small": swin_small,
            "swin_base": swin_base,
            "swin_large": swin_large,
        }
        if variant not in builders:
            raise ValueError(
                f"Unsupported Swin variant '{variant}'. "
                f"Choose one of {tuple(builders.keys())}."
            )

        builder = builders[variant]
        if pretrained:
            import lucid.weights as W

            weight_prefix = {
                "swin_tiny": "Swin_Tiny_Weights",
                "swin_base": "Swin_Base_Weights",
            }
            if variant not in weight_prefix:
                raise ValueError(
                    f"Pretrained backbone is not available for '{variant}' in lucid weights."
                )
            weights = getattr(W, weight_prefix[variant]).DEFAULT
            model = builder(
                img_size=image_size,
                num_classes=0,
                in_channels=in_channels,
                windows_size=window_size,
                drop_path_rate=drop_path_rate,
                weights=weights,
            )
        else:
            model = builder(
                img_size=image_size,
                num_classes=0,
                in_channels=in_channels,
                windows_size=window_size,
                drop_path_rate=drop_path_rate,
                weights=None,
            )

        self.variant = variant
        self.channels = self._CHANNELS_BY_VARIANT[variant]
        self.pretrained = pretrained

        # Keep naming close to HF backbone hierarchy where practical.
        self.embeddings = nn.Module()
        self.embeddings.patch_embeddings = nn.Module()
        self.embeddings.patch_embeddings.projection = model.patch_embed.proj
        self.embeddings.norm = (
            model.patch_embed.norm
            if model.patch_embed.norm is not None
            else nn.Identity()
        )

        self.abs_pos_emb = model.abs_pos_emb
        if self.abs_pos_emb:
            self.absolute_pos_emb = model.absolute_pos_emb

        self.pos_drop = model.pos_drop

        self.encoder = nn.Module()
        self.encoder.layers = model.layers

        self.hidden_states_norms = nn.Module()
        for idx, channels in enumerate(self.channels, start=1):
            self.hidden_states_norms.add_module(f"stage{idx}", nn.LayerNorm(channels))

        self.patches_res = tuple(model.patches_res)

    @classmethod
    def from_config(
        cls,
        config: Mask2FormerConfig,
        variant: str = "swin_small",
        pretrained: bool = False,
    ) -> "_Mask2FormerSwinBackbone":
        backbone_config = (
            config.backbone_config if isinstance(config.backbone_config, dict) else {}
        )
        image_size = int(backbone_config.get("image_size", 224))
        drop_path_rate = float(backbone_config.get("drop_path_rate", 0.3))
        window_size = int(backbone_config.get("window_size", 7))
        return cls(
            variant=variant,
            in_channels=config.num_channels,
            image_size=image_size,
            drop_path_rate=drop_path_rate,
            window_size=window_size,
            pretrained=pretrained,
        )

    def forward(self, pixel_values: Tensor) -> _BackboneOutput:
        hidden_states = self.embeddings.patch_embeddings.projection(pixel_values)
        batch_size = hidden_states.shape[0]
        height = hidden_states.shape[2]
        width = hidden_states.shape[3]
        hidden_states = hidden_states.reshape(batch_size, hidden_states.shape[1], -1)
        hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = self.embeddings.norm(hidden_states)
        if self.abs_pos_emb:
            hidden_states = hidden_states + self.absolute_pos_emb
        hidden_states = self.pos_drop(hidden_states)

        curr_height, curr_width = height, width
        feature_maps: list[Tensor] = []
        for stage_idx, layer in enumerate(self.encoder.layers):
            for block in layer.blocks:
                block.input_res = (curr_height, curr_width)
                hidden_states = block(hidden_states)

            norm = getattr(self.hidden_states_norms, f"stage{stage_idx + 1}")
            stage_hidden = norm(hidden_states)

            batch_size, _, channels = stage_hidden.shape
            feature = stage_hidden.transpose((0, 2, 1)).reshape(
                batch_size,
                channels,
                curr_height,
                curr_width,
            )
            feature_maps.append(feature)

            if layer.downsample is not None:
                layer.downsample.input_res = (curr_height, curr_width)
                hidden_states = layer.downsample(hidden_states)
                curr_height = (curr_height + 1) // 2
                curr_width = (curr_width + 1) // 2

        return _BackboneOutput(feature_maps=feature_maps)


def sample_point(
    input_features: Tensor,
    point_coordinates: Tensor,
    add_dim: bool = False,
    **kwargs,
) -> Tensor:
    if point_coordinates.ndim == 3:
        add_dim = True
        point_coordinates = point_coordinates.unsqueeze(axis=2)

    point_features = F.grid_sample(
        input_features,
        2.0 * point_coordinates - 1.0,
        mode=kwargs.get("mode", "bilinear"),
        padding_mode=kwargs.get("padding_mode", "zeros"),
        align_corners=kwargs.get("align_corners", False),
    )
    if add_dim:
        point_features = point_features.squeeze(axis=3)

    return point_features


def dice_loss(inputs: Tensor, labels: Tensor, num_masks: Tensor) -> Tensor:
    probs = F.sigmoid(inputs).flatten(start_axis=1)
    numerator = 2 * (probs * labels).sum(axis=-1)
    denominator = probs.sum(axis=-1) + labels.sum(axis=-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    loss = loss.sum() / num_masks
    return loss


def sigmoid_cross_entropy_loss(
    inputs: Tensor, labels: Tensor, num_masks: Tensor
) -> Tensor:
    cross_entropy_loss = F.binary_cross_entropy_with_logits(
        inputs, labels, reduction=None
    )
    loss = cross_entropy_loss.mean(axis=1).sum() / num_masks
    return loss


def pair_wise_dice_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    inputs = F.sigmoid(inputs).flatten(start_axis=1)
    numerator = 2 * lucid.matmul(inputs, labels.T)
    denominator = inputs.sum(axis=-1)[:, None] + labels.sum(axis=-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def pair_wise_sigmoid_cross_entropy_loss(inputs: Tensor, labels: Tensor) -> Tensor:
    height_and_width = inputs.shape[1]

    cross_entropy_loss_pos = F.binary_cross_entropy_with_logits(
        inputs,
        lucid.ones_like(inputs),
        reduction=None,
    )
    cross_entropy_loss_neg = F.binary_cross_entropy_with_logits(
        inputs,
        lucid.zeros_like(inputs),
        reduction=None,
    )

    loss_pos = lucid.matmul(cross_entropy_loss_pos / height_and_width, labels.T)
    loss_neg = lucid.matmul(cross_entropy_loss_neg / height_and_width, (1 - labels).T)
    loss = loss_pos + loss_neg
    return loss


class _Mask2FormerHungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 1.0,
        cost_mask: float = 1.0,
        cost_dice: float = 1.0,
        num_points: int = 12544,
    ) -> None:
        super().__init__()
        if cost_class == 0 and cost_mask == 0 and cost_dice == 0:
            raise ValueError("All costs can't be 0")

        self.num_points = num_points
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

        batch_size = masks_queries_logits.shape[0]
        for i in range(batch_size):
            labels = class_labels[i]
            if labels.size == 0:
                empty = lucid.empty(
                    0, dtype=lucid.Int32, device=masks_queries_logits.device
                )
                indices.append((empty, empty))
                continue

            pred_probs = F.softmax(class_queries_logits[i], axis=-1)
            pred_mask = masks_queries_logits[i]

            cost_class = -pred_probs[:, labels]
            target_mask = mask_labels[i].astype(pred_mask.dtype)
            target_mask = target_mask[:, None]
            pred_mask = pred_mask[:, None]

            point_coordinates = lucid.random.rand(
                (1, self.num_points, 2),
                device=pred_mask.device,
            )

            target_coordinates = point_coordinates.repeat(target_mask.shape[0], axis=0)
            target_mask = sample_point(
                target_mask,
                target_coordinates,
                align_corners=False,
            ).squeeze(axis=1)

            pred_coordinates = point_coordinates.repeat(pred_mask.shape[0], axis=0)
            pred_mask = sample_point(
                pred_mask,
                pred_coordinates,
                align_corners=False,
            ).squeeze(axis=1)

            cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
            cost_dice = pair_wise_dice_loss(pred_mask, target_mask)

            cost_matrix = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            )
            max_val = Tensor(1e10, dtype=cost_matrix.dtype, device=cost_matrix.device)
            min_val = Tensor(-1e10, dtype=cost_matrix.dtype, device=cost_matrix.device)
            cost_matrix = lucid.minimum(cost_matrix, max_val)
            cost_matrix = lucid.maximum(cost_matrix, min_val)

            assigned_indices = linear_sum_assignment(cost_matrix)
            indices.append(assigned_indices)

        matched_indices = [
            (i.astype(lucid.Int32), j.astype(lucid.Int32)) for i, j in indices
        ]
        return matched_indices


class _Mask2FormerLoss(nn.Module):
    def __init__(
        self, config: Mask2FormerConfig, weight_dict: dict[str, float]
    ) -> None:
        super().__init__()
        self.num_labels = config.num_labels
        self.weight_dict = weight_dict

        self.eos_coef = config.no_object_weight
        empty_weight = lucid.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.num_points = config.train_num_points
        self.oversample_ratio = config.oversample_ratio
        self.importance_sample_ratio = config.importance_sample_ratio

        self.matcher = _Mask2FormerHungarianMatcher(
            cost_class=config.class_weight,
            cost_dice=config.dice_weight,
            cost_mask=config.mask_weight,
            num_points=self.num_points,
        )

    def _max_by_axis(self, sizes: list[list[int]]) -> list[int]:
        maxes = sizes[0]
        for sublist in sizes[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    def _pad_images_to_max_in_batch(
        self, tensors: list[Tensor]
    ) -> tuple[Tensor, Tensor]:
        max_size = self._max_by_axis([list(tensor.shape) for tensor in tensors])
        batch_shape = [len(tensors)] + max_size
        batch_size, _, height, width = batch_shape

        dtype = tensors[0].dtype
        device = tensors[0].device
        padded_tensors = lucid.zeros(batch_shape, dtype=dtype, device=device)
        padding_masks = lucid.ones(
            (batch_size, height, width), dtype=bool, device=device
        )

        for i, tensor in enumerate(tensors):
            c, h, w = tensor.shape
            padded_tensors[i, :c, :h, :w] = tensor
            padding_masks[i, :h, :w] = False

        return padded_tensors, padding_masks

    def _get_predictions_permutation_indices(
        self,
        indices: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor]:
        batch_indices: list[Tensor] = []
        prediction_indices: list[Tensor] = []
        device = indices[0][0].device if len(indices) > 0 else self.empty_weight.device

        for i, (src, _) in enumerate(indices):
            if src.size == 0:
                continue
            batch_indices.append(
                lucid.full(src.shape, i, dtype=lucid.Int32, device=device)
            )
            prediction_indices.append(src.astype(lucid.Int32))

        if not prediction_indices:
            return (
                lucid.empty(0, dtype=lucid.Int32, device=device),
                lucid.empty(0, dtype=lucid.Int32, device=device),
            )

        return (
            lucid.concatenate(tuple(batch_indices), axis=0),
            lucid.concatenate(tuple(prediction_indices), axis=0),
        )

    def _get_targets_permutation_indices(
        self,
        indices: list[tuple[Tensor, Tensor]],
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
            -1,
            pred_logits_transposed.shape[1],
        )
        target_classes_flat = target_classes.reshape(-1)

        loss_ce = F.cross_entropy(
            pred_logits_flat,
            target_classes_flat,
            weight=self.empty_weight,
        )
        return {"loss_cross_entropy": loss_ce}

    def calculate_uncertainty(self, logits: Tensor) -> Tensor:
        return -lucid.abs(logits)

    def sample_points_using_uncertainty(
        self,
        logits: Tensor,
        uncertainty_function: Callable[[Tensor], Tensor],
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
    ) -> Tensor:
        num_boxes = logits.shape[0]
        num_points_sampled = int(num_points * oversample_ratio)

        point_coordinates = lucid.random.rand(
            (num_boxes, num_points_sampled, 2),
            device=logits.device,
        )
        point_logits = sample_point(logits, point_coordinates, align_corners=False)
        point_uncertainties = uncertainty_function(point_logits)

        num_uncertain_points = int(importance_sample_ratio * num_points)
        num_random_points = num_points - num_uncertain_points

        if num_uncertain_points > 0:
            uncertain_coords = []
            for b in range(num_boxes):
                _, idx = lucid.topk(
                    point_uncertainties[b, 0],
                    k=num_uncertain_points,
                    axis=0,
                )
                uncertain_coords.append(point_coordinates[b, idx.astype(lucid.Int32)])
            point_coordinates = lucid.stack(tuple(uncertain_coords), axis=0)
        else:
            point_coordinates = lucid.empty(
                (num_boxes, 0, 2),
                dtype=logits.dtype,
                device=logits.device,
            )

        if num_random_points > 0:
            random_coords = lucid.random.rand(
                (num_boxes, num_random_points, 2),
                device=logits.device,
            )
            point_coordinates = lucid.concatenate(
                (point_coordinates, random_coords), axis=1
            )

        return point_coordinates

    def loss_masks(
        self,
        masks_queries_logits: Tensor,
        mask_labels: list[Tensor],
        indices: list[tuple[Tensor, Tensor]],
        num_masks: Tensor,
    ) -> dict[str, Tensor]:
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)

        if src_idx[0].size == 0:
            zero = lucid.zeros(
                (),
                dtype=masks_queries_logits.dtype,
                device=masks_queries_logits.device,
            )
            return {"loss_mask": zero, "loss_dice": zero}

        pred_masks = masks_queries_logits[src_idx]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx]

        pred_masks = pred_masks[:, None]
        target_masks = target_masks[:, None]

        with lucid.no_grad():
            point_coordinates = self.sample_points_using_uncertainty(
                pred_masks,
                self.calculate_uncertainty,
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            point_labels = sample_point(
                target_masks,
                point_coordinates,
                align_corners=False,
            ).squeeze(axis=1)

        point_logits = sample_point(
            pred_masks,
            point_coordinates,
            align_corners=False,
        ).squeeze(axis=1)

        losses = {
            "loss_mask": sigmoid_cross_entropy_loss(
                point_logits, point_labels, num_masks
            ),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }
        return losses

    def _compute_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
    ) -> dict[str, Tensor]:
        indices = self.matcher(
            masks_queries_logits,
            class_queries_logits,
            mask_labels,
            class_labels,
        )
        num_masks = self.get_num_masks(class_labels, device=class_labels[0].device)

        losses: dict[str, Tensor] = {
            **self.loss_masks(masks_queries_logits, mask_labels, indices, num_masks),
            **self.loss_labels(class_queries_logits, class_labels, indices),
        }
        return losses

    def forward(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        auxiliary_predictions: list[dict[str, Tensor]] | None = None,
    ) -> dict[str, Tensor]:
        losses = self._compute_loss_dict(
            masks_queries_logits,
            class_queries_logits,
            mask_labels,
            class_labels,
        )

        if auxiliary_predictions is not None:
            for idx, aux_outputs in enumerate(auxiliary_predictions):
                aux_loss = self._compute_loss_dict(
                    aux_outputs["masks_queries_logits"],
                    aux_outputs["class_queries_logits"],
                    mask_labels,
                    class_labels,
                )
                aux_loss = {f"{key}_{idx}": value for key, value in aux_loss.items()}
                losses.update(aux_loss)

        return losses


def multi_scale_deformable_attention(
    value: Tensor,
    value_spatial_shapes: list[tuple[int, int]],
    sampling_locations: Tensor,
    attention_weights: Tensor,
) -> Tensor:
    batch_size, _, num_heads, hidden_dim = value.shape
    _, num_queries, _, num_levels, num_points, _ = sampling_locations.shape

    value_list: list[Tensor] = []
    start = 0
    for height, width in value_spatial_shapes:
        end = start + height * width
        value_list.append(value[:, start:end])
        start = end

    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []

    for level_id, (height, width) in enumerate(value_spatial_shapes):
        value_l = value_list[level_id]
        value_l = value_l.transpose((0, 2, 3, 1)).reshape(
            batch_size * num_heads,
            hidden_dim,
            height,
            width,
        )

        sampling_grid_l = sampling_grids[:, :, :, level_id]
        sampling_grid_l = sampling_grid_l.transpose((0, 2, 1, 3, 4)).reshape(
            batch_size * num_heads,
            num_queries,
            num_points,
            2,
        )

        sampling_value_l = F.grid_sample(
            value_l,
            sampling_grid_l,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l)

    attention_weights = attention_weights.transpose((0, 2, 1, 3, 4)).reshape(
        batch_size * num_heads,
        1,
        num_queries,
        num_levels * num_points,
    )

    stacked = lucid.stack(tuple(sampling_value_list), axis=-2).reshape(
        batch_size * num_heads,
        hidden_dim,
        num_queries,
        num_levels * num_points,
    )

    output = (
        (stacked * attention_weights)
        .sum(axis=-1)
        .reshape(
            batch_size,
            num_heads * hidden_dim,
            num_queries,
        )
    )
    return output.transpose((0, 2, 1))


class _Mask2FormerSinePositionEmbedding(nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(
        self,
        shape: tuple[int, ...],
        device: str,
        dtype: Any,
        mask: Tensor | None = None,
    ) -> Tensor:
        if mask is None:
            mask = lucid.zeros(
                (shape[0], shape[2], shape[3]), device=device, dtype=bool
            )

        not_mask = (~mask).astype(dtype)
        y_embed = lucid.cumsum(not_mask, axis=1)
        x_embed = lucid.cumsum(not_mask, axis=2)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = lucid.arange(
            self.num_pos_feats, dtype=lucid.Int64, device=device
        ).astype(dtype)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t

        pos_x = lucid.stack(
            (lucid.sin(pos_x[..., 0::2]), lucid.cos(pos_x[..., 1::2])),
            axis=4,
        ).flatten(start_axis=3)
        pos_y = lucid.stack(
            (lucid.sin(pos_y[..., 0::2]), lucid.cos(pos_y[..., 1::2])),
            axis=4,
        ).flatten(start_axis=3)

        pos = lucid.concatenate((pos_y, pos_x), axis=3).transpose((0, 3, 1, 2))
        return pos


class _Mask2FormerPixelMultiscaleDeformableAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, n_levels: int, n_points: int
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim (d_model) must be divisible by num_heads, "
                f"but got {embed_dim} and {num_heads}"
            )

        self.im2col_step = 128
        self.d_model = embed_dim
        self.n_levels = n_levels
        self.n_heads = num_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(
            embed_dim,
            num_heads * n_levels * n_points * 2,
        )
        self.attention_weights = nn.Linear(
            embed_dim,
            num_heads * n_levels * n_points,
        )
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def with_pos_embed(
        self, tensor: Tensor, position_embeddings: Tensor | None
    ) -> Tensor:
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        encoder_hidden_states: Tensor | None = None,
        encoder_attention_mask: Tensor | None = None,
        position_embeddings: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        del encoder_attention_mask, level_start_index

        if (
            encoder_hidden_states is None
            or reference_points is None
            or spatial_shapes_list is None
        ):
            raise ValueError(
                "encoder_hidden_states, reference_points, and spatial_shapes_list "
                "must be provided."
            )

        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        batch_size, num_queries, _ = hidden_states.shape
        _, sequence_length, _ = encoder_hidden_states.shape

        total_elements = sum(height * width for height, width in spatial_shapes_list)
        if total_elements != sequence_length:
            raise ValueError(
                "Make sure spatial_shapes_list aligns with sequence length. "
                f"Expected {total_elements}, got {sequence_length}."
            )

        value = self.value_proj(encoder_hidden_states)
        if attention_mask is not None:
            value = value.masked_fill(attention_mask[..., None], float(0))

        value = value.reshape(
            batch_size, sequence_length, self.n_heads, self.d_model // self.n_heads
        )

        sampling_offsets = self.sampling_offsets(hidden_states).reshape(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
            2,
        )

        attention_weights = self.attention_weights(hidden_states).reshape(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels * self.n_points,
        )
        attention_weights = F.softmax(attention_weights, axis=-1).reshape(
            batch_size,
            num_queries,
            self.n_heads,
            self.n_levels,
            self.n_points,
        )

        if reference_points.shape[-1] == 2:
            offset_normalizer = Tensor(
                [[shape[1], shape[0]] for shape in spatial_shapes_list],
                dtype=reference_points.dtype,
                device=reference_points.device,
            )
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.n_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, got {reference_points.shape[-1]}"
            )

        output = multi_scale_deformable_attention(
            value,
            spatial_shapes_list,
            sampling_locations,
            attention_weights,
        )
        output = self.output_proj(output)

        return output, (attention_weights if output_attentions else None)


class _Mask2FormerPixelDecoderEncoderLayer(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__()
        self.embed_dim = config.feature_size
        self.self_attn = _Mask2FormerPixelMultiscaleDeformableAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            n_levels=3,
            n_points=4,
        )

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout

        activation_fn = nn.utils.get_activation_from_name(config.activation_function)
        if activation_fn is None:
            raise ValueError(
                f"Invalid activation function: '{config.activation_function}'"
            )
        self.activation_fn: Callable[[Tensor], Tensor] = activation_fn

        self.activation_dropout = config.dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_feedforward_dim)
        self.fc2 = nn.Linear(config.encoder_feedforward_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        position_embeddings: Tensor | None = None,
        reference_points: Tensor | None = None,
        spatial_shapes_list: list[tuple[int, int]] | None = None,
        level_start_index: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, ...]:
        residual = hidden_states

        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes_list=spatial_shapes_list,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

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
            outputs += (attn_weights,)
        return outputs


class _Mask2FormerPixelDecoderEncoderOnly(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__()
        self.config = config
        self.dropout = config.dropout
        self.layers = nn.ModuleList(
            [
                _Mask2FormerPixelDecoderEncoderLayer(config)
                for _ in range(config.encoder_layers)
            ]
        )

    @staticmethod
    def get_reference_points(
        spatial_shapes_list: list[tuple[int, int]],
        valid_ratios: Tensor,
        device: str,
    ) -> Tensor:
        reference_points_list = []

        for lvl, (height, width) in enumerate(spatial_shapes_list):
            y_coords = lucid.linspace(
                0.5,
                height - 0.5,
                height,
                dtype=valid_ratios.dtype,
                device=device,
            )
            x_coords = lucid.linspace(
                0.5,
                width - 0.5,
                width,
                dtype=valid_ratios.dtype,
                device=device,
            )
            ref_y = y_coords[:, None].repeat(width, axis=1)
            ref_x = x_coords[None, :].repeat(height, axis=0)

            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = lucid.stack((ref_x, ref_y), axis=-1)
            reference_points_list.append(ref)

        reference_points = lucid.concatenate(tuple(reference_points_list), axis=1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        position_embeddings: Tensor,
        spatial_shapes_list: list[tuple[int, int]],
        level_start_index: Tensor,
        valid_ratios: Tensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...] | None]:
        del level_start_index

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

        hidden_states = inputs_embeds
        reference_points = self.get_reference_points(
            spatial_shapes_list,
            valid_ratios,
            device=inputs_embeds.device,
        )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        for encoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states.transpose((1, 0, 2)),)

            layer_outputs = encoder_layer(
                hidden_states,
                attention_mask,
                position_embeddings=position_embeddings,
                reference_points=reference_points,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states.transpose((1, 0, 2)),)

        return {
            "last_hidden_state": hidden_states,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }


class _Mask2FormerPixelDecoder(nn.Module):
    def __init__(self, config: Mask2FormerConfig, feature_channels: list[int]) -> None:
        super().__init__()

        self.config = config

        feature_dim = config.feature_size
        mask_dim = config.mask_feature_size
        num_pos_features = feature_dim // 2

        self.position_embedding = _Mask2FormerSinePositionEmbedding(
            num_pos_feats=num_pos_features,
            normalize=True,
        )
        self.num_feature_levels = 3
        transformer_in_channels = feature_channels[-self.num_feature_levels :]

        self.transformer_feature_strides = config.feature_strides[
            -self.num_feature_levels :
        ]
        self.feature_channels = feature_channels
        self.level_embed = nn.Parameter(
            lucid.empty((self.num_feature_levels, feature_dim))
        )

        if self.num_feature_levels > 1:
            input_projections_list = []
            for in_channels in transformer_in_channels[::-1]:
                input_projections_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, feature_dim, kernel_size=1),
                        nn.GroupNorm(32, feature_dim),
                    )
                )
            self.input_projections = nn.ModuleList(input_projections_list)
        else:
            self.input_projections = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(
                            transformer_in_channels[-1], feature_dim, kernel_size=1
                        ),
                        nn.GroupNorm(32, feature_dim),
                    )
                ]
            )

        self.encoder = _Mask2FormerPixelDecoderEncoderOnly(config)
        self.mask_projection = nn.Conv2d(
            feature_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        stride = min(self.transformer_feature_strides)
        self.common_stride = config.common_stride
        self.num_fpn_levels = int(math.log2(stride) - math.log2(self.common_stride))

        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(self.feature_channels[: self.num_fpn_levels]):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, feature_dim, kernel_size=1, bias=False),
                nn.GroupNorm(32, feature_dim),
            )
            output_conv = nn.Sequential(
                nn.Conv2d(
                    feature_dim,
                    feature_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(32, feature_dim),
                nn.ReLU(),
            )

            self.add_module(f"adapter_{idx + 1}", lateral_conv)
            self.add_module(f"layer_{idx + 1}", output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convolutions = lateral_convs[::-1]
        self.output_convolutions = output_convs[::-1]

    def get_valid_ratio(self, mask: Tensor, dtype: Any = lucid.Float32) -> Tensor:
        _, height, width = mask.shape

        valid_height = (~mask[:, :, 0]).sum(axis=1)
        valid_width = (~mask[:, 0, :]).sum(axis=1)

        valid_ratio_height = valid_height.astype(dtype) / height
        valid_ratio_width = valid_width.astype(dtype) / width

        valid_ratio = lucid.stack((valid_ratio_width, valid_ratio_height), axis=-1)
        return valid_ratio

    def forward(
        self,
        features: list[Tensor],
        encoder_outputs: dict[str, Any] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
    ) -> dict[str, Tensor | tuple[Tensor, ...] | None]:
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

        inputs_embeds = []
        position_embeddings = []
        for level, feature in enumerate(features[::-1][: self.num_feature_levels]):
            inputs_embeds.append(self.input_projections[level](feature))
            position_embeddings.append(
                self.position_embedding(feature.shape, feature.device, feature.dtype)
            )

        masks = [
            lucid.zeros(
                (x.shape[0], x.shape[2], x.shape[3]), device=x.device, dtype=bool
            )
            for x in inputs_embeds
        ]

        spatial_shapes_list = [
            (embed.shape[2], embed.shape[3]) for embed in inputs_embeds
        ]

        input_embeds_flat = lucid.concatenate(
            tuple(
                embed.flatten(start_axis=2).transpose((0, 2, 1))
                for embed in inputs_embeds
            ),
            axis=1,
        )

        masks_flat = lucid.concatenate(
            tuple(mask.flatten(start_axis=1) for mask in masks), axis=1
        )

        position_embeddings = [
            embed.flatten(start_axis=2).transpose((0, 2, 1))
            for embed in position_embeddings
        ]
        level_pos_embed_flat = [
            x + self.level_embed[i].reshape(1, 1, -1)
            for i, x in enumerate(position_embeddings)
        ]
        level_pos_embed_flat = lucid.concatenate(tuple(level_pos_embed_flat), axis=1)

        level_start_index_list = [0]
        for height, width in spatial_shapes_list[:-1]:
            level_start_index_list.append(level_start_index_list[-1] + height * width)

        level_start_index = Tensor(
            level_start_index_list,
            dtype=lucid.Int64,
            device=input_embeds_flat.device,
        )

        valid_ratios = lucid.stack(
            tuple(
                self.get_valid_ratio(mask, dtype=input_embeds_flat.dtype)
                for mask in masks
            ),
            axis=1,
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                inputs_embeds=input_embeds_flat,
                attention_mask=masks_flat,
                position_embeddings=level_pos_embed_flat,
                spatial_shapes_list=spatial_shapes_list,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

        last_hidden_state = encoder_outputs["last_hidden_state"]
        batch_size = last_hidden_state.shape[0]

        encoder_output = []
        start = 0
        for height, width in spatial_shapes_list:
            end = start + height * width
            encoder_output.append(last_hidden_state[:, start:end])
            start = end

        outputs = [
            x.transpose((0, 2, 1)).reshape(
                batch_size, -1, spatial_shapes_list[i][0], spatial_shapes_list[i][1]
            )
            for i, x in enumerate(encoder_output)
        ]

        for idx, feature in enumerate(features[: self.num_fpn_levels][::-1]):
            lateral_conv = self.lateral_convolutions[idx]
            output_conv = self.output_convolutions[idx]
            current_fpn = lateral_conv(feature)
            out = current_fpn + F.interpolate(
                outputs[-1],
                size=current_fpn.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            out = output_conv(out)
            outputs.append(out)

        multi_scale_features = []
        num_cur_levels = 0
        for out in outputs:
            if num_cur_levels < self.num_feature_levels:
                multi_scale_features.append(out)
                num_cur_levels += 1

        return {
            "mask_features": self.mask_projection(outputs[-1]),
            "multi_scale_features": tuple(multi_scale_features),
            "attentions": encoder_outputs["attentions"],
        }


class _Mask2FormerPixelLevelModule(nn.Module):
    def __init__(self, config: Mask2FormerConfig, backbone: nn.Module) -> None:
        super().__init__()
        self.encoder = backbone
        self.decoder = _Mask2FormerPixelDecoder(
            config, feature_channels=self.encoder.channels
        )

    def forward(
        self,
        pixel_values: Tensor,
        output_hidden_states: bool = False,
    ) -> dict[str, Tensor | tuple[Tensor, ...] | None]:
        backbone_features = self.encoder(pixel_values).feature_maps
        decoder_output = self.decoder(
            backbone_features,
            output_hidden_states=output_hidden_states,
        )

        return {
            "encoder_last_hidden_state": backbone_features[-1],
            "encoder_hidden_states": (
                tuple(backbone_features) if output_hidden_states else None
            ),
            "decoder_last_hidden_state": decoder_output["mask_features"],
            "decoder_hidden_states": decoder_output["multi_scale_features"],
        }


class _Mask2FormerAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        del is_decoder

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(
                "embed_dim must be divisible by num_heads "
                f"(got embed_dim={self.embed_dim}, num_heads={num_heads})."
            )

        self.scaling = self.head_dim**-0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: Tensor, seq_len: int, batch_size: int) -> Tensor:
        return tensor.reshape(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose((0, 2, 1, 3))

    def with_pos_embed(
        self, tensor: Tensor, position_embeddings: Tensor | None
    ) -> Tensor:
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        position_embeddings: Tensor | None = None,
        key_value_states: Tensor | None = None,
        key_value_position_embeddings: Tensor | None = None,
        output_attentions: bool = False,
    ) -> tuple[Tensor, Tensor | None]:
        hidden_states = (
            hidden_states.transpose((1, 0, 2)) if hidden_states is not None else None
        )
        position_embeddings = (
            position_embeddings.transpose((1, 0, 2))
            if position_embeddings is not None
            else None
        )
        key_value_states = (
            key_value_states.transpose((1, 0, 2))
            if key_value_states is not None
            else None
        )
        key_value_position_embeddings = (
            key_value_position_embeddings.transpose((1, 0, 2))
            if key_value_position_embeddings is not None
            else None
        )

        is_cross_attention = key_value_states is not None
        batch_size, target_len, embed_dim = hidden_states.shape

        hidden_states_original = hidden_states
        if position_embeddings is not None:
            hidden_states = self.with_pos_embed(hidden_states, position_embeddings)

        key_value_states_original = key_value_states
        if key_value_position_embeddings is not None:
            key_value_states = self.with_pos_embed(
                key_value_states,
                key_value_position_embeddings,
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
                "Attention weights should be of size "
                f"{(batch_size * self.num_heads, target_len, source_len)}, "
                f"but is {attn_weights.shape}"
            )

        if attention_mask is not None:
            if attention_mask.shape != (
                batch_size * self.num_heads,
                target_len,
                source_len,
            ):
                raise ValueError(
                    "Attention mask should be of size "
                    f"{(batch_size * self.num_heads, target_len, source_len)}, "
                    f"but is {attention_mask.shape}"
                )
            if attention_mask.dtype is bool:
                attention_mask = lucid.zeros_like(
                    attention_mask, dtype=attn_weights.dtype
                ).masked_fill(
                    attention_mask,
                    _MASK_ATTENTION_FILL_VALUE,
                )
            attn_weights = attn_weights + attention_mask

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

        attn_output = attn_output.reshape(
            batch_size, self.num_heads, target_len, self.head_dim
        )
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(
            batch_size, target_len, embed_dim
        )

        attn_output = self.out_proj(attn_output).transpose((1, 0, 2))
        return attn_output, attn_weights_reshaped


class _Mask2FormerMaskedAttentionDecoderLayer(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_dim
        self.pre_norm = config.pre_norm

        self.self_attn = _Mask2FormerAttention(
            embed_dim=self.embed_dim,
            num_heads=config.num_attention_heads,
            dropout=config.dropout,
            is_decoder=True,
        )

        self.dropout = config.dropout

        activation_fn = nn.utils.get_activation_from_name(config.activation_function)
        if activation_fn is None:
            raise ValueError(
                f"Invalid activation function: '{config.activation_function}'"
            )
        self.activation_fn: Callable[[Tensor], Tensor] = activation_fn

        self.activation_dropout = config.dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn = nn.MultiHeadAttention(
            self.embed_dim,
            config.num_attention_heads,
            config.dropout,
            use_separate_proj_weight=False,
        )
        self.cross_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.fc1 = nn.Linear(self.embed_dim, config.dim_feedforward)
        self.fc2 = nn.Linear(config.dim_feedforward, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def _format_cross_attention_mask(
        self,
        attention_mask: Tensor | None,
        batch_size: int,
        target_len: int,
        source_len: int,
        dtype: Any,
    ) -> Tensor | None:
        if attention_mask is None:
            return None

        if (
            attention_mask.ndim == 3
            and attention_mask.shape[0] == batch_size * self.config.num_attention_heads
        ):
            attention_mask = attention_mask.reshape(
                batch_size,
                self.config.num_attention_heads,
                target_len,
                source_len,
            )
        elif attention_mask.ndim == 3 and attention_mask.shape[0] == batch_size:
            attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.ndim == 4:
            pass
        else:
            raise ValueError(
                "Unsupported encoder_attention_mask shape for cross-attention: "
                f"{attention_mask.shape}."
            )

        if attention_mask.dtype is bool:
            attention_mask = lucid.zeros_like(attention_mask, dtype=dtype).masked_fill(
                attention_mask,
                _MASK_ATTENTION_FILL_VALUE,
            )

        return attention_mask

    def forward_post(
        self,
        hidden_states: Tensor,
        level_index: int | None = None,
        attention_mask: Tensor | None = None,
        position_embeddings: list[Tensor] | None = None,
        query_position_embeddings: Tensor | None = None,
        encoder_hidden_states: list[Tensor] | None = None,
        encoder_attention_mask: Tensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[Tensor, ...]:
        del attention_mask

        if (
            level_index is None
            or position_embeddings is None
            or encoder_hidden_states is None
        ):
            raise ValueError(
                "level_index, position_embeddings, and encoder_hidden_states are required."
            )

        residual = hidden_states

        query = self.with_pos_embed(hidden_states, query_position_embeddings)
        key = self.with_pos_embed(
            encoder_hidden_states[level_index],
            position_embeddings[level_index],
        )
        value = encoder_hidden_states[level_index]

        batch_size = query.shape[1]
        target_len = query.shape[0]
        source_len = key.shape[0]

        cross_mask = self._format_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            target_len,
            source_len,
            dtype=query.dtype,
        )

        hidden_states = self.cross_attn(
            query.transpose((1, 0, 2)),
            key.transpose((1, 0, 2)),
            value.transpose((1, 0, 2)),
            attn_mask=cross_mask,
            key_padding_mask=None,
        ).transpose((1, 0, 2))

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

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
            outputs += (self_attn_weights, None)

        return outputs

    def forward_pre(
        self,
        hidden_states: Tensor,
        level_index: int | None = None,
        attention_mask: Tensor | None = None,
        position_embeddings: list[Tensor] | None = None,
        query_position_embeddings: Tensor | None = None,
        encoder_hidden_states: list[Tensor] | None = None,
        encoder_attention_mask: Tensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[Tensor, ...]:
        del attention_mask

        if (
            level_index is None
            or position_embeddings is None
            or encoder_hidden_states is None
        ):
            raise ValueError(
                "level_index, position_embeddings, and encoder_hidden_states are required."
            )

        residual = hidden_states
        hidden_states = self.cross_attn_layer_norm(hidden_states)

        query = self.with_pos_embed(hidden_states, query_position_embeddings)
        key = self.with_pos_embed(
            encoder_hidden_states[level_index],
            position_embeddings[level_index],
        )
        value = encoder_hidden_states[level_index]

        batch_size = query.shape[1]
        target_len = query.shape[0]
        source_len = key.shape[0]

        cross_mask = self._format_cross_attention_mask(
            encoder_attention_mask,
            batch_size,
            target_len,
            source_len,
            dtype=query.dtype,
        )

        hidden_states = self.cross_attn(
            query.transpose((1, 0, 2)),
            key.transpose((1, 0, 2)),
            value.transpose((1, 0, 2)),
            attn_mask=cross_mask,
            key_padding_mask=None,
        ).transpose((1, 0, 2))

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=query_position_embeddings,
            attention_mask=None,
            output_attentions=True,
        )

        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = F.dropout(
            hidden_states, p=self.activation_dropout, training=self.training
        )
        hidden_states = self.fc2(hidden_states)
        hidden_states = F.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, None)

        return outputs

    def forward(
        self,
        hidden_states: Tensor,
        level_index: int | None = None,
        attention_mask: Tensor | None = None,
        position_embeddings: list[Tensor] | None = None,
        query_position_embeddings: Tensor | None = None,
        encoder_hidden_states: list[Tensor] | None = None,
        encoder_attention_mask: Tensor | None = None,
        output_attentions: bool | None = False,
    ) -> tuple[Tensor, ...]:
        if self.pre_norm:
            return self.forward_pre(
                hidden_states=hidden_states,
                level_index=level_index,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                query_position_embeddings=query_position_embeddings,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )

        return self.forward_post(
            hidden_states=hidden_states,
            level_index=level_index,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            query_position_embeddings=query_position_embeddings,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
        )


class _Mask2FormerPredictionBlock(nn.Module):
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


class _Mask2FormerMLPPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        in_dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        out_dims = [hidden_dim] * (num_layers - 1) + [output_dim]

        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            activation = nn.ReLU() if i < num_layers - 1 else nn.Identity()
            layer = _Mask2FormerPredictionBlock(in_dim, out_dim, activation=activation)
            self.layers.append(layer)
            self.add_module(str(i), layer)

    def forward(self, input_: Tensor) -> Tensor:
        hidden_state = input_
        for layer in self.layers:
            hidden_state = layer(hidden_state)
        return hidden_state


class _Mask2FormerMaskPredictor(nn.Module):
    def __init__(
        self, hidden_size: int, num_heads: int, mask_feature_size: int
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.mask_embedder = _Mask2FormerMLPPredictionHead(
            self.hidden_size,
            self.hidden_size,
            mask_feature_size,
        )

    def forward(
        self,
        outputs: Tensor,
        pixel_embeddings: Tensor,
        attention_mask_target_size: tuple[int, int] | None = None,
    ) -> tuple[Tensor, Tensor]:
        mask_embeddings = self.mask_embedder(outputs.transpose((1, 0, 2)))

        outputs_mask = lucid.einops.einsum(
            "bqc,bchw->bqhw",
            mask_embeddings,
            pixel_embeddings,
        )

        attention_mask = F.interpolate(
            outputs_mask,
            size=attention_mask_target_size,
            mode="bilinear",
            align_corners=False,
        )

        attention_mask = F.sigmoid(attention_mask).flatten(start_axis=2)
        attention_mask = attention_mask.unsqueeze(axis=1).repeat(self.num_heads, axis=1)
        attention_mask = (
            attention_mask.reshape(
                attention_mask.shape[0] * attention_mask.shape[1],
                attention_mask.shape[2],
                attention_mask.shape[3],
            )
            < 0.5
        )

        return outputs_mask, attention_mask.detach()


class _Mask2FormerMaskedAttentionDecoder(nn.Module):
    def __init__(self, config: Mask2FormerConfig) -> None:
        super().__init__()
        self.config = config
        self.mask_feature_size = config.mask_feature_size
        self.dropout = config.dropout
        self.layerdrop = config.dropout
        self.num_feature_levels = 3
        self.decoder_layers = config.decoder_layers - 1

        self.layers = nn.ModuleList(
            [
                _Mask2FormerMaskedAttentionDecoderLayer(config)
                for _ in range(self.decoder_layers)
            ]
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim)

        self.mask_predictor = _Mask2FormerMaskPredictor(
            hidden_size=config.hidden_dim,
            num_heads=config.num_attention_heads,
            mask_feature_size=self.mask_feature_size,
        )

    def forward(
        self,
        inputs_embeds: Tensor | None = None,
        multi_stage_positional_embeddings: list[Tensor] | None = None,
        pixel_embeddings: Tensor | None = None,
        encoder_hidden_states: list[Tensor] | None = None,
        query_position_embeddings: Tensor | None = None,
        feature_size_list: list[tuple[int, int]] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
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

        if (
            inputs_embeds is None
            or multi_stage_positional_embeddings is None
            or pixel_embeddings is None
            or encoder_hidden_states is None
            or feature_size_list is None
        ):
            raise ValueError(
                "inputs_embeds, multi_stage_positional_embeddings, pixel_embeddings, "
                "encoder_hidden_states, and feature_size_list must be provided."
            )

        hidden_states = inputs_embeds

        intermediate = ()
        all_hidden_states = () if output_hidden_states else None
        attentions = () if output_attentions else None
        intermediate_mask_predictions = ()

        intermediate_hidden_states = self.layernorm(inputs_embeds)
        intermediate += (intermediate_hidden_states,)

        predicted_mask, attention_mask = self.mask_predictor(
            intermediate_hidden_states,
            pixel_embeddings,
            feature_size_list[0],
        )
        intermediate_mask_predictions += (predicted_mask,)

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = lucid.random.rand(
                    (), device=hidden_states.device
                ).item()
                if dropout_probability < self.layerdrop:
                    continue

            level_index = idx % self.num_feature_levels

            valid_rows = attention_mask.sum(axis=-1) != attention_mask.shape[-1]
            attention_mask = attention_mask & valid_rows.unsqueeze(axis=-1)

            layer_outputs = decoder_layer(
                hidden_states,
                level_index,
                None,
                multi_stage_positional_embeddings,
                query_position_embeddings,
                encoder_hidden_states,
                encoder_attention_mask=attention_mask,
                output_attentions=output_attentions,
            )

            intermediate_hidden_states = self.layernorm(layer_outputs[0])
            predicted_mask, attention_mask = self.mask_predictor(
                intermediate_hidden_states,
                pixel_embeddings,
                feature_size_list[(idx + 1) % self.num_feature_levels],
            )

            intermediate_mask_predictions += (predicted_mask,)
            intermediate += (intermediate_hidden_states,)

            hidden_states = layer_outputs[0]
            if output_attentions:
                attentions += (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return {
            "last_hidden_state": hidden_states.transpose((1, 0, 2)),
            "hidden_states": all_hidden_states,
            "attentions": attentions,
            "intermediate_hidden_states": intermediate,
            "masks_queries_logits": intermediate_mask_predictions,
        }


class _Mask2FormerTransformerModule(nn.Module):
    def __init__(self, in_features: int, config: Mask2FormerConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim

        self.num_feature_levels = 3
        self.position_embedder = _Mask2FormerSinePositionEmbedding(
            num_pos_feats=hidden_dim // 2,
            normalize=True,
        )

        self.queries_embedder = nn.Embedding(config.num_queries, hidden_dim)
        self.queries_features = nn.Embedding(config.num_queries, hidden_dim)

        self.input_projections: list[nn.Module] = []
        for _ in range(self.num_feature_levels):
            if in_features != hidden_dim or config.enforce_input_projection:
                self.input_projections.append(
                    nn.Conv2d(in_features, hidden_dim, kernel_size=1)
                )
            else:
                self.input_projections.append(nn.Sequential())

        self.decoder = _Mask2FormerMaskedAttentionDecoder(config=config)
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)

    def forward(
        self,
        multi_scale_features: list[Tensor] | tuple[Tensor, ...],
        mask_features: Tensor,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
    ) -> dict[str, Any]:
        multi_stage_features = []
        multi_stage_positional_embeddings = []
        size_list = []

        for i in range(self.num_feature_levels):
            size_list.append(multi_scale_features[i].shape[-2:])
            multi_stage_positional_embeddings.append(
                self.position_embedder(
                    multi_scale_features[i].shape,
                    multi_scale_features[i].device,
                    multi_scale_features[i].dtype,
                    None,
                ).flatten(start_axis=2)
            )

            multi_stage_features.append(
                self.input_projections[i](multi_scale_features[i]).flatten(start_axis=2)
                + self.level_embed.weight[i][None, :, None]
            )

            multi_stage_positional_embeddings[-1] = multi_stage_positional_embeddings[
                -1
            ].transpose((2, 0, 1))
            multi_stage_features[-1] = multi_stage_features[-1].transpose((2, 0, 1))

        _, batch_size, _ = multi_stage_features[0].shape

        query_embeddings = self.queries_embedder.weight.unsqueeze(axis=1).repeat(
            batch_size, axis=1
        )
        query_features = self.queries_features.weight.unsqueeze(axis=1).repeat(
            batch_size, axis=1
        )

        decoder_output = self.decoder(
            inputs_embeds=query_features,
            multi_stage_positional_embeddings=multi_stage_positional_embeddings,
            pixel_embeddings=mask_features,
            encoder_hidden_states=multi_stage_features,
            query_position_embeddings=query_embeddings,
            feature_size_list=size_list,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        return decoder_output


class _Mask2FormerModel(nn.Module):
    def __init__(
        self, config: Mask2FormerConfig, backbone: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.config = config

        if backbone is None:
            resnet_variant = _maskformer_infer_resnet_variant(config.backbone_config)
            swin_variant = _mask2former_infer_swin_variant(config.backbone_config)

            if resnet_variant is not None:
                backbone = _MaskFormerResNetBackbone.from_config(
                    config=config,
                    variant=resnet_variant,
                    pretrained=False,
                )
            elif swin_variant is not None:
                backbone = _Mask2FormerSwinBackbone.from_config(
                    config=config,
                    variant=swin_variant,
                    pretrained=False,
                )
            else:
                raise ValueError(
                    "backbone must be provided to _Mask2FormerModel when "
                    "backbone_config is missing or unsupported."
                )

        self.pixel_level_module = _Mask2FormerPixelLevelModule(config, backbone)
        self.transformer_module = _Mask2FormerTransformerModule(
            in_features=config.feature_size,
            config=config,
        )

        if getattr(self.pixel_level_module.encoder, "pretrained", False):
            for module in self.pixel_level_module.encoder.modules():
                module._skip_mask2former_init = True

        self.apply(self._init_weights)

    @lucid.no_grad()
    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "_skip_mask2former_init", False):
            return

        xavier_std = self.config.init_xavier_std
        std = self.config.init_std

        if isinstance(module, _Mask2FormerTransformerModule):
            if module.input_projections is not None:
                for input_projection in module.input_projections:
                    if not isinstance(input_projection, nn.Sequential):
                        input_projection._skip_default_init = True
                        nn.init.xavier_uniform(input_projection.weight, gain=xavier_std)
                        if input_projection.bias is not None:
                            nn.init.constant(input_projection.bias, 0.0)

        elif isinstance(module, _Mask2FormerPixelMultiscaleDeformableAttention):
            nn.init.constant(module.sampling_offsets.weight, 0.0)

            thetas = lucid.arange(module.n_heads, dtype=lucid.Float32)
            thetas = thetas * (2.0 * math.pi / module.n_heads)
            grid_init = lucid.stack((lucid.cos(thetas), lucid.sin(thetas)), axis=-1)
            grid_init = grid_init / lucid.max(
                lucid.abs(grid_init), axis=-1, keepdims=True
            )
            grid_init = grid_init.reshape(module.n_heads, 1, 1, 2)
            grid_init = grid_init.repeat(module.n_levels, axis=1)
            grid_init = grid_init.repeat(module.n_points, axis=2)

            for i in range(module.n_points):
                grid_init[:, :, i, :] *= i + 1

            module.sampling_offsets.bias.data = grid_init.reshape(-1).data

            nn.init.constant(module.attention_weights.weight, 0.0)
            nn.init.constant(module.attention_weights.bias, 0.0)
            nn.init.xavier_uniform(module.value_proj.weight)
            nn.init.constant(module.value_proj.bias, 0.0)
            nn.init.xavier_uniform(module.output_proj.weight)
            nn.init.constant(module.output_proj.bias, 0.0)

        elif isinstance(module, _Mask2FormerMaskedAttentionDecoderLayer):
            for p in module.parameters():
                if p.ndim > 1:
                    nn.init.xavier_uniform(p, gain=xavier_std)
            if module.cross_attn.in_proj_bias is not None:
                nn.init.constant(module.cross_attn.in_proj_bias, 0.0)

        elif isinstance(module, _Mask2FormerPixelDecoder):
            nn.init.normal(module.level_embed, mean=0.0, std=0.0)

        if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)) and not getattr(
            module,
            "_skip_default_init",
            False,
        ):
            nn.init.normal(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.constant(module.bias, 0.0)

            if getattr(module, "running_mean", None) is not None:
                nn.init.constant(module.running_mean, 0.0)
                nn.init.constant(module.running_var, 1.0)

        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            if module.weight is not None:
                nn.init.constant(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant(module.bias, 0.0)

        elif isinstance(module, nn.Embedding):
            nn.init.normal(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx] = 0

        elif isinstance(module, _Mask2FormerLoss):
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
        del kwargs

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

        batch_size, _, height, width = pixel_values.shape
        if pixel_mask is None:
            pixel_mask = lucid.ones(
                (batch_size, height, width), device=pixel_values.device
            )

        pixel_level_module_output = self.pixel_level_module(
            pixel_values=pixel_values,
            output_hidden_states=output_hidden_states,
        )

        transformer_module_output = self.transformer_module(
            multi_scale_features=pixel_level_module_output["decoder_hidden_states"],
            mask_features=pixel_level_module_output["decoder_last_hidden_state"],
            output_hidden_states=True,
            output_attentions=output_attentions,
        )

        encoder_hidden_states = None
        pixel_decoder_hidden_states = None
        transformer_decoder_hidden_states = None
        transformer_decoder_intermediate_states = transformer_module_output[
            "intermediate_hidden_states"
        ]

        if output_hidden_states:
            encoder_hidden_states = pixel_level_module_output["encoder_hidden_states"]
            pixel_decoder_hidden_states = pixel_level_module_output[
                "decoder_hidden_states"
            ]
            transformer_decoder_hidden_states = transformer_module_output[
                "hidden_states"
            ]

        return {
            "encoder_last_hidden_state": pixel_level_module_output[
                "encoder_last_hidden_state"
            ],
            "pixel_decoder_last_hidden_state": pixel_level_module_output[
                "decoder_last_hidden_state"
            ],
            "transformer_decoder_last_hidden_state": transformer_module_output[
                "last_hidden_state"
            ],
            "encoder_hidden_states": encoder_hidden_states,
            "pixel_decoder_hidden_states": pixel_decoder_hidden_states,
            "transformer_decoder_hidden_states": transformer_decoder_hidden_states,
            "transformer_decoder_intermediate_states": transformer_decoder_intermediate_states,
            "masks_queries_logits": transformer_module_output["masks_queries_logits"],
            "attentions": transformer_module_output["attentions"],
        }


class Mask2Former(PreTrainedModelMixin, nn.Module):
    def __init__(
        self, config: Mask2FormerConfig, backbone: nn.Module | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.model = _Mask2FormerModel(config, backbone=backbone)

        self.weight_dict: dict[str, float] = {
            "loss_cross_entropy": config.class_weight,
            "loss_mask": config.mask_weight,
            "loss_dice": config.dice_weight,
        }

        self.class_predictor = nn.Linear(config.hidden_dim, config.num_labels + 1)
        self.criterion = _Mask2FormerLoss(config=config, weight_dict=self.weight_dict)

    def get_auxiliary_logits(
        self,
        classes: tuple[Tensor, ...],
        output_masks: tuple[Tensor, ...],
    ) -> list[dict[str, Tensor]]:
        auxiliary_logits = []
        for aux_binary_masks, aux_classes in zip(output_masks[:-1], classes[:-1]):
            auxiliary_logits.append(
                {
                    "masks_queries_logits": aux_binary_masks,
                    "class_queries_logits": aux_classes,
                }
            )
        return auxiliary_logits

    def get_loss_dict(
        self,
        masks_queries_logits: Tensor,
        class_queries_logits: Tensor,
        mask_labels: list[Tensor],
        class_labels: list[Tensor],
        auxiliary_predictions: list[dict[str, Tensor]] | None,
    ) -> dict[str, Tensor]:
        loss_dict: dict[str, Tensor] = self.criterion(
            masks_queries_logits=masks_queries_logits,
            class_queries_logits=class_queries_logits,
            mask_labels=mask_labels,
            class_labels=class_labels,
            auxiliary_predictions=auxiliary_predictions,
        )

        for key, weight in self.weight_dict.items():
            for loss_key in list(loss_dict.keys()):
                if key in loss_key:
                    loss_dict[loss_key] = loss_dict[loss_key] * weight

        return loss_dict

    def get_loss(self, loss_dict: dict[str, Tensor]) -> Tensor:
        return sum(loss_dict.values())

    def forward(
        self,
        pixel_values: Tensor,
        mask_labels: list[Tensor] | None = None,
        class_labels: list[Tensor] | None = None,
        pixel_mask: Tensor | None = None,
        output_hidden_states: bool | None = None,
        output_auxiliary_logits: bool | None = None,
        output_attentions: bool | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        del kwargs

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )

        if output_auxiliary_logits is None:
            if self.config.output_auxiliary_logits is not None:
                output_auxiliary_logits = self.config.output_auxiliary_logits
            else:
                output_auxiliary_logits = self.config.use_auxiliary_loss

        raw_outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
        )

        transformer_decoder_intermediate_states = raw_outputs[
            "transformer_decoder_intermediate_states"
        ]
        masks_queries_logits = raw_outputs["masks_queries_logits"]

        if transformer_decoder_intermediate_states is None:
            raise ValueError(
                "transformer_decoder_intermediate_states is required for Mask2Former heads."
            )

        class_queries_logits = tuple(
            self.class_predictor(hidden_state.transpose((1, 0, 2)))
            for hidden_state in transformer_decoder_intermediate_states
        )

        auxiliary_logits = None
        auxiliary_predictions = None
        if self.config.use_auxiliary_loss:
            auxiliary_predictions = self.get_auxiliary_logits(
                class_queries_logits,
                masks_queries_logits,
            )
            if output_auxiliary_logits:
                auxiliary_logits = auxiliary_predictions

        loss = None
        if mask_labels is not None and class_labels is not None:
            loss_dict = self.get_loss_dict(
                masks_queries_logits=masks_queries_logits[-1],
                class_queries_logits=class_queries_logits[-1],
                mask_labels=mask_labels,
                class_labels=class_labels,
                auxiliary_predictions=auxiliary_predictions,
            )
            loss = self.get_loss(loss_dict)

        return {
            **raw_outputs,
            "loss": loss,
            "class_queries_logits": class_queries_logits[-1],
            "masks_queries_logits": masks_queries_logits[-1],
            "auxiliary_logits": auxiliary_logits,
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

        original_hw = (pixel_values.shape[-2], pixel_values.shape[-1])
        if pixel_mask is None:
            stride = (
                max(self.config.feature_strides) if self.config.feature_strides else 32
            )
            pad_h = (stride - original_hw[0] % stride) % stride
            pad_w = (stride - original_hw[1] % stride) % stride

            if pad_h or pad_w:
                pixel_values = lucid.pad(
                    pixel_values,
                    ((0, 0), (0, 0), (0, pad_h), (0, pad_w)),
                )

            pixel_mask = lucid.zeros(
                (pixel_values.shape[0], pixel_values.shape[-2], pixel_values.shape[-1]),
                dtype=bool,
                device=pixel_values.device,
            )
            if pad_h or pad_w:
                pixel_mask[:, original_hw[0] :, :] = True
                pixel_mask[:, :, original_hw[1] :] = True

        outputs = self.forward(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=False,
            output_attentions=False,
            output_auxiliary_logits=False,
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
            output_size = original_hw

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


def _mask2former_preset_config(variant: str, num_labels: int) -> Mask2FormerConfig:
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

    backbone_config = {
        "model_type": "resnet",
        "depths": base_depths[variant],
        "hidden_sizes": base_hidden_sizes[variant],
    }

    return Mask2FormerConfig(
        num_labels=num_labels,
        feature_size=256,
        mask_feature_size=256,
        hidden_dim=256,
        backbone_config=backbone_config,
        num_queries=100,
        encoder_layers=6,
        encoder_feedforward_dim=1024,
        decoder_layers=10,
        dim_feedforward=2048,
        num_attention_heads=8,
        feature_strides=[4, 8, 16, 32],
        common_stride=4,
        enforce_input_projection=False,
        activation_function="relu",
        pre_norm=False,
        dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        dilation=False,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1,
        train_num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        use_auxiliary_loss=True,
    )


def _mask2former_swin_preset_config(variant: str, num_labels: int) -> Mask2FormerConfig:
    backbone_params = {
        "swin_tiny": {
            "embed_dim": 96,
            "depths": [2, 2, 6, 2],
            "num_heads": [3, 6, 12, 24],
            "hidden_sizes": [96, 192, 384, 768],
            "image_size": 224,
            "window_size": 7,
        },
        "swin_small": {
            "embed_dim": 96,
            "depths": [2, 2, 18, 2],
            "num_heads": [3, 6, 12, 24],
            "hidden_sizes": [96, 192, 384, 768],
            "image_size": 224,
            "window_size": 7,
        },
        "swin_base": {
            "embed_dim": 128,
            "depths": [2, 2, 18, 2],
            "num_heads": [4, 8, 16, 32],
            "hidden_sizes": [128, 256, 512, 1024],
            "image_size": 384,
            "window_size": 12,
        },
        "swin_large": {
            "embed_dim": 192,
            "depths": [2, 2, 18, 2],
            "num_heads": [6, 12, 24, 48],
            "hidden_sizes": [192, 384, 768, 1536],
            "image_size": 384,
            "window_size": 12,
        },
    }

    if variant not in backbone_params:
        raise ValueError(
            f"Unsupported variant '{variant}'. "
            f"Choose one of {tuple(backbone_params.keys())}."
        )

    params = backbone_params[variant]
    backbone_config = {
        "model_type": "swin",
        "image_size": params["image_size"],
        "patch_size": 4,
        "num_channels": 3,
        "embed_dim": params["embed_dim"],
        "depths": params["depths"],
        "num_heads": params["num_heads"],
        "window_size": params["window_size"],
        "drop_path_rate": 0.3,
        "out_features": ["stage1", "stage2", "stage3", "stage4"],
        "out_indices": [1, 2, 3, 4],
        "hidden_sizes": params["hidden_sizes"],
    }

    return Mask2FormerConfig(
        num_labels=num_labels,
        feature_size=256,
        mask_feature_size=256,
        hidden_dim=256,
        backbone_config=backbone_config,
        num_queries=100,
        encoder_layers=6,
        encoder_feedforward_dim=1024,
        decoder_layers=10,
        dim_feedforward=2048,
        num_attention_heads=8,
        feature_strides=[4, 8, 16, 32],
        common_stride=4,
        enforce_input_projection=False,
        activation_function="relu",
        pre_norm=False,
        dropout=0.0,
        init_std=0.02,
        init_xavier_std=1.0,
        dilation=False,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        no_object_weight=0.1,
        train_num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        use_auxiliary_loss=True,
    )


def _mask2former_config_to_kwargs(config: Mask2FormerConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for field_name in Mask2FormerConfig.__dataclass_fields__.keys():
        kwargs[field_name] = copy.deepcopy(getattr(config, field_name))
    return kwargs


def _build_mask2former_with_resnet_backbone(
    variant: str,
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    config = _mask2former_preset_config(variant=variant, num_labels=num_labels)
    merged_kwargs = _mask2former_config_to_kwargs(config)
    merged_kwargs.update(config_kwargs)
    merged_kwargs["num_labels"] = num_labels
    config = Mask2FormerConfig(**merged_kwargs)

    backbone = _MaskFormerResNetBackbone.from_config(
        config=config,
        variant=variant,
        pretrained=pretrained_backbone,
    )
    return Mask2Former(config=config, backbone=backbone)


def _build_mask2former_with_swin_backbone(
    variant: str,
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    config = _mask2former_swin_preset_config(variant=variant, num_labels=num_labels)
    merged_kwargs = _mask2former_config_to_kwargs(config)
    merged_kwargs.update(config_kwargs)
    merged_kwargs["num_labels"] = num_labels
    config = Mask2FormerConfig(**merged_kwargs)

    backbone = _Mask2FormerSwinBackbone.from_config(
        config=config,
        variant=variant,
        pretrained=pretrained_backbone,
    )
    return Mask2Former(config=config, backbone=backbone)


def _build_mask2former_with_variant(
    variant: str,
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    if variant.startswith("resnet_"):
        return _build_mask2former_with_resnet_backbone(
            variant=variant,
            num_labels=num_labels,
            pretrained_backbone=pretrained_backbone,
            **config_kwargs,
        )
    if variant.startswith("swin_"):
        return _build_mask2former_with_swin_backbone(
            variant=variant,
            num_labels=num_labels,
            pretrained_backbone=pretrained_backbone,
            **config_kwargs,
        )

    raise ValueError(
        f"Unsupported Mask2Former variant '{variant}'. "
        "Expected one of resnet_* or swin_*."
    )


@register_model
def mask2former_resnet_18(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_resnet_backbone(
        "resnet_18",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def mask2former_resnet_34(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_resnet_backbone(
        "resnet_34",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def mask2former_resnet_50(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_resnet_backbone(
        "resnet_50",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def mask2former_resnet_101(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_resnet_backbone(
        "resnet_101",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def mask2former_swin_tiny(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_swin_backbone(
        "swin_tiny",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def mask2former_swin_small(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_swin_backbone(
        "swin_small",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def mask2former_swin_base(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_swin_backbone(
        "swin_base",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )


@register_model
def mask2former_swin_large(
    num_labels: int,
    *,
    pretrained_backbone: bool = False,
    **config_kwargs: Any,
) -> Mask2Former:
    return _build_mask2former_with_swin_backbone(
        "swin_large",
        num_labels=num_labels,
        pretrained_backbone=pretrained_backbone,
        **config_kwargs,
    )
