import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _DeviceType

from .fast_rcnn import _SlowROIPool
from ._util import apply_deltas, nms, clip_boxes


__all__ = ["FasterRCNN"]


class _AnchorGenerator(nn.Module):
    def __init__(
        self, sizes: tuple[int, ...], ratios: tuple[float, ...], stride: int
    ) -> None:
        super().__init__()
        self.stride = stride
        base_anchors = []
        for size in sizes:
            area = float(size * size)
            for r in ratios:
                w = math.sqrt(area / r)
                h = w * r
                base_anchors.append([-w / 2, -h / 2, w / 2, h / 2])

        self.base_anchors = Tensor(base_anchors, dtype=lucid.Float32)

    @property
    def num_anchors(self) -> int:
        return len(self.base_anchors)

    def grid_anchors(self, feat_h: int, feat_w: int, device: _DeviceType) -> Tensor:
        shift_x = (lucid.arange(feat_w, device=device) + 0.5) * self.stride
        shift_y = (lucid.arange(feat_h, device=device) + 0.5) * self.stride

        y, x = lucid.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = lucid.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)

        anchors = self.base_anchors.to(device).reshape(1.0 - 1, 4)
        anchors += shifts.reshape(-1, 1, 4)

        return anchors.reshape(-1, 4)


class _RPNHead(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = F.relu(self.conv(x))
        logits = self.cls_logits(x)
        deltas = self.bbox_pred(x)

        return logits, deltas


class _RegionProposalNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        anchor_generator: _AnchorGenerator,
        pre_nms_top_n: int = 6000,
        post_nms_top_n: int = 1000,
        nms_thresh: float = 0.7,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.head = _RPNHead(in_channels, anchor_generator.num_anchors)
        self.anchor_generator = anchor_generator
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_thresh = nms_thresh
        self.score_thresh = score_thresh

    def forward(self, feature: Tensor, image_shape: tuple[int, int]) -> list[Tensor]:
        B, _, H, W = feature.shape
        logits, deltas = self.head(feature)
        logits = logits.transpose((0, 2, 3, 1)).reshape(B, -1)
        deltas = deltas.transpose((0, 2, 3, 1)).reshape(B, -1, 4)

        anchors = self.anchor_generator.grid_anchors(H, W, feature.device)
        proposals: list[Tensor] = []

        for b in range(B):
            scores = F.sigmoid(logits[b])
            boxes = apply_deltas(anchors, deltas[b], add_one=1.0)
            boxes = clip_boxes(boxes, image_shape)

            keep = scores > self.score_thresh
            if lucid.sum(keep) == 0:
                proposals.append(lucid.empty(0, 4, device=feature.device))
                continue

            scores = scores[keep]
            boxes = boxes[keep]
            order = lucid.argsort(scores, descending=True)
            if self.pre_nms_top_n:
                order = order[: self.pre_nms_top_n]

            boxes = boxes[order]
            scores = scores[order]

            keep_idx = nms(boxes, scores, self.nms_thresh)
            if self.post_nms_top_n:
                keep_idx = keep_idx[: self.post_nms_top_n]

            proposals.append(boxes[keep_idx])

        return proposals


class FasterRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_channels: int,
        num_classes: int,
        *,
        anchor_sizes: tuple[int, ...] = (128, 256, 512),
        aspect_ratios: tuple[int, ...] = (0.5, 1.0, 2.0),
        anchor_stride: int = 16,
        pool_size: tuple[int, int] = (7, 7),
        hidden_dim: int = 4096,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.anchor_generator = _AnchorGenerator(
            anchor_sizes, aspect_ratios, anchor_stride
        )
        self.rpn = _RegionProposalNetwork(feat_channels, self.anchor_generator)
        self.roipool = _SlowROIPool(pool_size)

        fc_in = feat_channels * pool_size[0] * pool_size[1]
        self.fc1 = nn.Linear(fc_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)

        self.bbox_reg_means = Tensor([0.0, 0.0, 0.0, 0.0], dtype=lucid.Float32)
        self.bbox_reg_stds = Tensor([0.1, 0.1, 0.2, 0.2], dtype=lucid.Float32)

    def forward(
        self,
        images: Tensor,
        rois: Tensor | None = None,
        roi_idx: Tensor | None = None,
        *,
        return_feats: bool = False,
    ) -> tuple[Tensor, ...]:
        H, W = images.shape[2:]
        feats = self.backbone(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        if rois is None or roi_idx is None:
            proposals = self.rpn(feats, (H, W))
            boxes_list, idx_list = [], []
            for i, p in enumerate(proposals):
                if p.size == 0:
                    continue
                boxes_list.append(p)
                idx_list.append(
                    lucid.full(
                        (p.shape[0],), i, dtype=lucid.Int32, device=images.device
                    )
                )

            if boxes_list:
                rois_px = lucid.concatenate(boxes_list, axis=0)
                roi_idx = lucid.concatenate(idx_list, axis=0)
            else:
                rois_px = lucid.empty(0, 4, device=images.device)
                roi_idx = lucid.empty(0, dtype=lucid.Int32, device=images.device)

            rois = rois_px / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
            rois.to(images.device)

        else:
            rois_px = rois * lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
            rois_px.to(images.device)

        pooled = self.roipool(feats, rois, roi_idx)
        N = pooled.shape[0]

        x = pooled.reshape(N, -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        if return_feats:
            return cls_logits, bbox_deltas, feats
        return cls_logits, bbox_deltas

    @lucid.no_grad()
    def predict(
        self,
        images: Tensor,
        *,
        score_thresh: float = 0.05,
        nms_thresh: float = 0.5,
        top_k: int = 100,
    ) -> list[dict[str, Tensor]]:
        B, _, H, W = images.shape
        feats = self.backbone(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        proposals = self.rpn(feats, (H, W))
        boxes_list, idx_list = [], []
        for i, p in enumerate(proposals):
            if p.size == 0:
                continue
            boxes_list.append(p)
            idx_list.append(
                lucid.full((p.shape[0],), i, dtype=lucid.Int32, device=images.device)
            )
        if boxes_list:
            rois_px = lucid.concatenate(boxes_list, axis=0)
            roi_idx = lucid.concatenate(idx_list, axis=0)
        else:
            rois_px = lucid.empty(0, 4, device=images.device)
            roi_idx = lucid.empty(0, dtype=lucid.Int32, device=images.device)

        rois_norm = rois_px / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
        rois_norm.to(images.device)

        cls_logits, bbox_deltas = self.forward(images, rois_norm, roi_idx)
        scores = F.softmax(cls_logits, axis=1)

        num_classes = scores.shape[1]
        detections = [{"boxes": [], "scores": [], "labels": []} for _ in range(B)]

        for c in range(1, num_classes):
            cls_scores = scores[:, c]
            deltas_cls = bbox_deltas[:, c * 4 : (c + 1) * 4]

            boxes_all = apply_deltas(rois_px, deltas_cls)
            boxes_all = clip_boxes(boxes_all, (H, W))

            mask = cls_scores > score_thresh
            if lucid.sum(mask) == 0:
                continue
            cls_scores = cls_scores[mask]
            boxes = boxes_all[mask]
            img_ids = roi_idx[mask]

            for img_id in img_ids.unique():
                m = img_ids == img_id
                boxes_i = boxes[m]
                scores_i = cls_scores[m]

                keep = nms(boxes_i, scores_i, nms_thresh)[:top_k]
                if keep.size == 0:
                    continue

                det = detections[int(img_id.item())]
                det["boxes"].append(boxes_i[keep])
                det["scores"].append(scores_i[keep])
                det["labels"].append(
                    lucid.full((keep.size,), c, dtype=lucid.Int32, device=images.device)
                )

        for det in detections:
            if det["boxes"]:
                det["boxes"] = lucid.concatenate(det["boxes"], axis=0)
                det["scores"] = lucid.concatenate(det["scores"], axis=0)
                det["labels"] = lucid.concatenate(det["labels"], axis=0)
            else:
                det["boxes"] = lucid.empty(0, 4, device=images.device)
                det["scores"] = lucid.empty(0, device=images.device)
                det["labels"] = lucid.empty(0, dtype=lucid.Int32, device=images.device)

        return detections
