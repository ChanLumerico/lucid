from typing import Callable
import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

from lucid.models.objdet.util import (
    ROIAlign,
    SelectiveSearch,
    apply_deltas,
    nms,
    clip_boxes,
)


__all__ = ["FastRCNN"]


class FastRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_channels: int,
        num_classes: int,
        pool_size: tuple[int, int] = (7, 7),
        hidden_dim: int = 4096,
        bbox_reg_means: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0),
        bbox_reg_stds: tuple[float, ...] = (0.1, 0.1, 0.2, 0.2),
        dropout: float = 0.5,
        proposal_generator: Callable[..., Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.roipool = ROIAlign(output_size=pool_size)
        self.proposal_generator = proposal_generator or SelectiveSearch()

        self.fc1 = nn.Linear(feat_channels * pool_size[0] * pool_size[1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

        self.bbox_reg_means = Tensor(bbox_reg_means, dtype=lucid.Float32)
        self.bbox_reg_stds = Tensor(bbox_reg_stds, dtype=lucid.Float32)

    def forward(
        self,
        images: Tensor,
        rois: Tensor | None = None,
        roi_idx: Tensor | None = None,
        *,
        return_feats: bool = False
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
                start = c * 4
                preds.append(deltas_fg[i, start : start + 4])

            preds = lucid.stack(preds, axis=0)
            targets_fg = targets_norm[fg_mask]
            reg_loss = F.huber_loss(preds, targets_fg)
        else:
            reg_loss = lucid.zeros((), dtype=lucid.Float32)

        total_loss = cls_loss + reg_loss
        return total_loss, cls_loss, reg_loss
