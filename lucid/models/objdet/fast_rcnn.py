from typing import Callable
import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor

from ._util import SelectiveSearch, apply_deltas, nms, clip_boxes


class _SlowROIPool(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size)
        self.output_size = output_size

    def forward(self, images, rois, roi_idx):
        N = rois.shape[0]
        H, W = images.shape[2:]

        x1, x2 = rois[:, 0], rois[:, 2]
        y1, y2 = rois[:, 1], rois[:, 3]

        x1 = lucid.floor(x1 * W).astype(lucid.Int)
        x2 = lucid.ceil(x2 * W).astype(lucid.Int)
        y1 = lucid.floor(y1 * H).astype(lucid.Int)
        y2 = lucid.ceil(y2 * H).astype(lucid.Int)

        res = []
        for i in range(N):
            img = images[roi_idx[i]].unsqueeze(axis=0)
            img = img[:, :, y1[i] : y2[i], x1[i] : x2[i]]
            img = self.maxpool(img)
            res.append(img)

        res = lucid.concatenate(res, axis=0)
        return res


class FastRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_channels: int,
        num_classes: int,
        pool_size: tuple[int, int] = (7, 7),
        hidden_dim: int = 4096,
        bbox_reg_means: tuple[int, ...] = (0.0, 0.0, 0.0, 0.0),
        bbox_reg_stds: tuple[int, ...] = (0.1, 0.1, 0.2, 0.2),
        dropout: float = 0.5,
        proposal_generator: Callable[..., Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.roipool = _SlowROIPool(output_size=pool_size)
        self.proposal_generator = proposal_generator or SelectiveSearch()

        self.fc1 = nn.Linear(feat_channels * pool_size[0] * pool_size[1], hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

        self.bbox_reg_means = Tensor(bbox_reg_means, dtype=lucid.Float16)
        self.bbox_reg_stds = Tensor(bbox_reg_stds, dtype=lucid.Float16)

    def forward(
        self,
        images: Tensor,
        rois: Tensor | None,
        roi_idx: Tensor | None,
        *,
        return_feats: bool = False
    ) -> tuple[Tensor, ...]:
        B, _, H, W = images.shape
        if rois is None or roi_idx is None:
            boxes_list, idx_list = [], []
            for i in range(B):
                props = self.proposal_generator(images[i])
                norm = props.astype(lucid.Float32) / lucid.Tensor(
                    [W, H, W, H], dtype=lucid.Float32
                )

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
        rois: Tensor,
        cls_logits: Tensor,
        bbox_deltas: Tensor,
        image_shape: tuple[int, int],
        score_thresh: float = 0.05,
        nms_thresh: float = 0.3,
        top_k: int = 100,
    ) -> list[dict[str, Tensor]]:
        scores = F.softmax(cls_logits, axis=1)
        num_classes = scores.shape[1]
        detections: list[dict[str, Tensor]] = []

        for cl in range(1, num_classes):
            cls_scores = scores[:, cl]
            mask = cls_scores > score_thresh
            if lucid.sum(mask) == 0:
                continue

            deltas_cls = bbox_deltas[:, cl * 4 : (cl + 1) * 4]
            boxes_all = apply_deltas(rois, deltas_cls)
            boxes = clip_boxes(boxes_all, image_shape)[mask]

            scores_masked = cls_scores[mask]
            keep = nms(boxes, scores_masked, nms_thresh)
            keep = keep[:top_k]

            detections.append(
                {
                    "boxes": boxes[keep],
                    "scores": scores_masked[keep],
                    "labels": lucid.full((keep.shape[0],), cl, dtype=lucid.Int16),
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
