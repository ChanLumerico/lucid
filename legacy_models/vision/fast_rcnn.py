from dataclasses import dataclass
from typing import Callable
import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

from lucid.models.utils import (
    ROIAlign,
    SelectiveSearch,
    apply_deltas,
    nms,
    clip_boxes,
)

__all__ = ["FastRCNN", "FastRCNNConfig"]


@dataclass
class FastRCNNConfig:
    backbone: nn.Module
    feat_channels: int
    num_classes: int
    pool_size: tuple[int, int] | list[int] = (7, 7)
    hidden_dim: int = 4096
    bbox_reg_means: tuple[float, ...] | list[float] = (0.0, 0.0, 0.0, 0.0)
    bbox_reg_stds: tuple[float, ...] | list[float] = (0.1, 0.1, 0.2, 0.2)
    dropout: float = 0.5
    proposal_generator: Callable[..., Tensor] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.backbone, nn.Module):
            raise TypeError("backbone must be an nn.Module")
        if self.feat_channels <= 0:
            raise ValueError("feat_channels must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")

        self.pool_size = tuple(self.pool_size)
        self.bbox_reg_means = tuple(self.bbox_reg_means)
        self.bbox_reg_stds = tuple(self.bbox_reg_stds)

        if len(self.pool_size) != 2 or self.pool_size[0] <= 0 or self.pool_size[1] <= 0:
            raise ValueError("pool_size must contain exactly two positive integers")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than 0")
        if len(self.bbox_reg_means) != 4:
            raise ValueError("bbox_reg_means must contain exactly four values")
        if len(self.bbox_reg_stds) != 4 or any(std <= 0 for std in self.bbox_reg_stds):
            raise ValueError("bbox_reg_stds must contain exactly four positive values")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.proposal_generator is not None and not callable(
            self.proposal_generator
        ):
            raise TypeError("proposal_generator must be callable or None")


class FastRCNN(nn.Module):
    def __init__(self, config: FastRCNNConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = config.backbone
        self.roipool = ROIAlign(output_size=config.pool_size)
        self.proposal_generator = config.proposal_generator or SelectiveSearch()

        self.fc1 = nn.Linear(
            config.feat_channels * config.pool_size[0] * config.pool_size[1],
            config.hidden_dim,
        )
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.cls_score = nn.Linear(config.hidden_dim, config.num_classes)
        self.bbox_pred = nn.Linear(config.hidden_dim, config.num_classes * 4)

        self.bbox_reg_means = Tensor(config.bbox_reg_means, dtype=lucid.Float32)
        self.bbox_reg_stds = Tensor(config.bbox_reg_stds, dtype=lucid.Float32)

    def forward(
        self,
        images: Tensor,
        rois: Tensor | None = None,
        roi_idx: Tensor | None = None,
        *,
        return_feats: bool = False,
    ) -> tuple[Tensor, ...]:
        B, _, H, W = images.shape
        if rois is None or roi_idx is None:
            boxes_list, idx_list = [], []
            for i in range(B):
                props = self.proposal_generator(images[i])
                props_f = props.astype(lucid.Float32)

                norm = props_f / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
                boxes_list.append(norm)
                idx_list.append(lucid.full((norm.shape[0],), i, dtype=lucid.Int32))

            rois = lucid.concatenate(boxes_list, axis=0)
            roi_idx = lucid.concatenate(idx_list, axis=0)

        feats = self.backbone(images)
        pooled = self.roipool(feats, rois, roi_idx)

        N = pooled.shape[0]
        x = pooled.reshape(N, -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        if return_feats:
            return cls_logits, bbox_deltas, feats
        return cls_logits, bbox_deltas

    @lucid.no_grad()
    def predict(
        self,
        images: Tensor,
        rois: Tensor | None = None,
        roi_idx: Tensor | None = None,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.3,
        top_k: int = 100,
    ) -> list[dict[str, Tensor]]:
        B, _, H, W = images.shape
        if rois is None or roi_idx is None:
            boxes_list, idx_list = [], []
            for i in range(B):
                props = self.proposal_generator(images[i])
                props_f = props.astype(lucid.Float32)
                boxes_list.append(props_f)
                idx_list.append(lucid.full((props_f.shape[0],), i, dtype=lucid.Int32))

            rois_px = lucid.concatenate(boxes_list, axis=0)
            roi_idx = lucid.concatenate(idx_list, axis=0)

            rois_norm = rois_px / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
            cls_logits, bbox_deltas = self.forward(images, rois_norm, roi_idx)
        else:
            rois_px = rois
            cls_logits, bbox_deltas = self.forward(images, rois, roi_idx)

        scores = F.softmax(cls_logits, axis=1)
        num_classes = scores.shape[1]
        detections: list[dict[str, Tensor]] = []

        for cl in range(1, num_classes):
            cls_scores = scores[:, cl]
            mask = cls_scores > score_thresh
            if lucid.sum(mask) == 0:
                continue

            deltas_cls = bbox_deltas[:, cl * 4 : (cl + 1) * 4]
            boxes_all = apply_deltas(rois_px, deltas_cls)
            boxes = clip_boxes(boxes_all, (H, W))[mask]

            scores_masked = cls_scores[mask]
            keep = nms(boxes, scores_masked, nms_thresh)[:top_k]

            detections.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores_masked[keep],
                    "labels": lucid.full((keep.shape[0],), cl, dtype=lucid.Int32),
                }
            )

        return detections

    def get_loss(
        self,
        cls_logits: Tensor,
        bbox_deltas: Tensor,
        labels: Tensor,
        reg_targets: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cls_loss = F.cross_entropy(cls_logits, labels)
        targets_norm = (reg_targets - self.bbox_reg_means) / self.bbox_reg_stds
        fg_mask = labels > 0

        if lucid.sum(fg_mask) > 0:
            labels_fg = labels[fg_mask]
            deltas_fg = bbox_deltas[fg_mask]

            preds: list[Tensor] = []
            for i, c in enumerate(labels_fg):
                start = int(c.item()) * 4
                preds.append(deltas_fg[i, start : start + 4])

            preds = lucid.stack(preds, axis=0)
            targets_fg = targets_norm[fg_mask]
            reg_loss = F.huber_loss(preds, targets_fg)
        else:
            reg_loss = lucid.zeros((), dtype=lucid.Float32)

        total_loss = cls_loss + reg_loss
        return total_loss, cls_loss, reg_loss
