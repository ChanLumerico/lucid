import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _DeviceType

from .fast_rcnn import _SlowROIPool  # Try vectorize this
from ._util import apply_deltas, nms, clip_boxes


__all__ = ["FasterRCNN"]

"""
NOTE

Implement
---------
1. `lucid.randperm` to lucid.random package.
2. add `.detach` functionality to `Tensor`

Further Improvements
--------------------
1. Vectorize ROIPool process; currently relies on pure Pythonic loops.

"""


def _box_iou(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    x1a, y1a, x2a, y2a = boxes_a.unbind(axis=1)
    x1b, y1b, x2b, y2b = boxes_b.unbind(axis=1)

    xx1 = lucid.maximum(x1a.unsqueeze(1), x1b.unsqueeze(0))
    yy1 = lucid.maximum(y1a.unsqueeze(1), y1b.unsqueeze(0))
    xx2 = lucid.minimum(x2a.unsqueeze(1), x2b.unsqueeze(0))
    yy2 = lucid.minimum(y2a.unsqueeze(1), y2b.unsqueeze(0))

    w = (xx2 - xx1 + 1).clip(min_value=0)
    h = (yy2 - yy1 + 1).clip(min_value=0)
    inter = w * h

    area_a = (x2a - x1a + 1) * (y2a - y1a + 1)
    area_b = (x2b - x1b + 1) * (y2b - y1b + 1)

    return inter / (area_a.unsqueeze(1) + area_b - inter)


def _bbox_to_delta(src: Tensor, target: Tensor, add_one: float = 1.0) -> Tensor:
    sw = src[:, 2] - src[:, 0] + add_one
    sh = src[:, 3] - src[:, 1] + add_one
    sx = src[:, 0] + 0.5 * sw
    sy = src[:, 1] + 0.5 * sh

    tw = target[:, 2] - target[:, 0] + add_one
    th = target[:, 3] - target[:, 1] + add_one
    tx = target[:, 0] + 0.5 * tw
    ty = target[:, 1] + 0.5 * th

    dx = (tx - sx) / sw
    dy = (ty - sy) / sh
    dw = lucid.log(tw / sw)
    dh = lucid.log(th / sh)

    return lucid.stack([dx, dy, dw, dh], axis=1)


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

        self.base_anchors: nn.Buffer
        self.register_buffer("base_anchors", Tensor(base_anchors, dtype=lucid.Float32))
        self._cache: dict[tuple[int, int, _DeviceType], Tensor] = {}

    @property
    def num_anchors(self) -> int:
        return len(self.base_anchors)

    def grid_anchors(self, feat_h: int, feat_w: int, device: _DeviceType) -> Tensor:
        key = (feat_h, feat_w, device)
        if key not in self._cache:
            self._cache.clear()

        if key in self._cache:
            return self._cache[key]

        shift_x = (lucid.arange(feat_w, device=device) + 0.5) * self.stride
        shift_y = (lucid.arange(feat_h, device=device) + 0.5) * self.stride

        y, x = lucid.meshgrid(shift_y, shift_x, indexing="ij")
        shifts = lucid.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)

        anchors = self.base_anchors.to(device).reshape(1, self.num_anchors, 4)
        anchors = anchors + shifts.reshape(-1, 1, 4)
        anchors = anchors.reshape(-1, 4)

        self._cache[key] = anchors
        return anchors


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
        aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
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

        self.bbox_reg_means: nn.Buffer
        self.bbox_reg_stds: nn.Buffer
        self.register_buffer(
            "bbox_reg_means", Tensor([0.0, 0.0, 0.0, 0.0], dtype=lucid.Float32)
        )
        self.register_buffer(
            "bbox_reg_stds", Tensor([0.1, 0.1, 0.2, 0.2], dtype=lucid.Float32)
        )

    def _roi_forward(
        self,
        feats: Tensor,
        rois: Tensor,
        roi_idx: Tensor,
        *,
        return_feats: bool = False,
    ) -> tuple[Tensor, ...]:
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
            rois = rois.to(images.device)
        else:
            rois_px = (rois * lucid.Tensor([W, H, W, H], dtype=lucid.Float32)).to(
                images.device
            )

        return self._roi_forward(feats, rois, roi_idx, return_feats=return_feats)

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
        rois_norm = rois_norm.to(images.device)

        cls_logits, bbox_deltas = self._roi_forward(feats, rois_norm, roi_idx)
        scores = F.softmax(cls_logits, axis=1)

        num_classes = scores.shape[1]
        detections = [{"boxes": [], "scores": [], "labels": []} for _ in range(B)]

        for c in range(1, num_classes):
            cls_scores = scores[:, c]
            deltas_cls = bbox_deltas[:, c * 4 : (c + 1) * 4]
            deltas_cls = deltas_cls * self.bbox_reg_stds + self.bbox_reg_means

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

    def get_loss(
        self, images: Tensor, targets: list[dict[str, Tensor]]
    ) -> dict[str, Tensor]:
        B, _, H, W = images.shape
        feats = self.backbone(images)
        if isinstance(feats, (tuple, list)):
            feats = feats[-1]

        Hf, Wf = feats.shape[2:]
        rpn_logits, rpn_deltas = self.rpn.head(feats)
        rpn_logits = rpn_logits.transpose((0, 2, 3, 1)).reshape(B, -1)
        rpn_deltas = rpn_deltas.transpose((0, 2, 3, 1)).reshape(B, -1, 4)

        anchors = self.anchor_generator.grid_anchors(Hf, Wf, images.device)

        rpn_cls_loss = lucid.zeros((), dtype=lucid.Float32)
        rpn_reg_loss = lucid.zeros((), dtype=lucid.Float32)
        proposals: list[Tensor] = []

        for b in range(B):
            gt_boxes = targets[b]["boxes"].to(images.device)

            if gt_boxes.size == 0:
                rpn_labels = lucid.zeros(
                    anchors.shape[0], dtype=lucid.Int32, device=images.device
                )
                rpn_reg_targets = lucid.zeros_like(anchors)
            else:
                ious = _box_iou(anchors, gt_boxes)
                max_iou = lucid.max(ious, axis=1)
                max_idx = lucid.argsort(ious, axis=1, descending=True)[:, 0]

                rpn_labels = lucid.full(
                    (anchors.shape[0],), -1, dtype=lucid.Int32, device=images.device
                )
                rpn_labels[max_iou < 0.3] = 0
                rpn_labels[max_iou >= 0.7] = 1

                for gi in range(gt_boxes.shape[0]):
                    best_anchor = lucid.argmax(ious[:, gi])
                    rpn_labels[best_anchor] = 1

                rpn_reg_targets = lucid.zeros(anchors.shape[0], 4, device=images.device)
                pos_mask = rpn_labels == 1

                if lucid.sum(pos_mask) > 0:
                    matched = gt_boxes[max_idx[pos_mask]]
                    rpn_reg_targets[pos_mask] = _bbox_to_delta(
                        anchors[pos_mask], matched
                    )

            pos_idx = (rpn_labels == 1).nonzero().squeeze(1)
            neg_idx = (rpn_labels == 0).nonzero().squeeze(1)

            num_pos = min(pos_idx.shape[0], 128)
            num_neg = min(neg_idx.shape[0], 256 - num_pos)

            # NOTE: Activate below after implementing `lucid.random.randperm`
            # perm_pos = lucid.randperm(pos_idx.shape[0])[:num_pos]
            # perm_neg = lucid.randperm(neg_idx.shape[0])[:num_neg]

            samp_idx = lucid.concatenate([pos_idx[perm_pos], neg_idx[perm_neg]], axis=0)

            cls_scores = F.sigmoid(rpn_logits[b][samp_idx])
            cls_targets = rpn_labels[samp_idx].astype(lucid.Float32)

            rpn_cls_loss += F.binary_cross_entropy(
                cls_scores, cls_targets, reduction="mean"
            )

            if num_pos > 0:
                reg_scores = rpn_deltas[b][pos_idx[perm_pos]]
                reg_targets = rpn_reg_targets[pos_idx[perm_pos]]
                rpn_reg_loss += F.huber_loss(reg_scores, reg_targets, reduction="mean")

            scores = F.sigmoid(rpn_logits[b])
            boxes = apply_deltas(anchors, rpn_deltas[b], add_one=1.0)
            boxes = clip_boxes(boxes, (H, W))

            keep = scores > self.rpn.score_thresh
            if lucid.sum(keep) == 0:
                proposals.append(lucid.empty(0, 4, device=images.device))
                continue

            scores = scores[keep]
            boxes = boxes[keep]
            order = lucid.argsort(scores, descending=True)
            if self.rpn.pre_nms_top_n:
                order = order[: self.rpn.pre_nms_top_n]

            boxes = boxes[order]
            scores = scores[order]

            keep_idx = nms(boxes, scores, self.rpn.nms_thresh)
            if self.rpn.post_nms_top_n:
                keep_idx = keep_idx[: self.rpn.post_nms_top_n]

            proposals.append(boxes[keep_idx])

        boxes_list, idx_list = [], []
        roi_labels_list, roi_reg_list = [], []
        for i, props in enumerate(proposals):
            if props.size == 0:
                continue

            boxes_list.append(props)
            idx_list.append(
                lucid.full(
                    (props.shape[0],), i, dtype=lucid.Int32, device=images.device
                )
            )
            gt_b = targets[i]["boxes"].to(images.device)
            gt_l = targets[i]["labels"].to(images.device)

            if gt_b.size == 0:
                roi_labels_list.append(
                    lucid.zeros(props.shape[0], dtype=lucid.Int32, device=images.device)
                )
                roi_reg_list.append(
                    lucid.zeros(props.shape[0], 4, device=images.device)
                )

            else:
                ious = _box_iou(props, gt_b)
                max_iou = lucid.max(ious, axis=1)
                max_idx = lucid.argsort(ious, axis=1, descending=True)[:, 0]

                labels = gt_l[max_idx].astype(lucid.Int32)
                labels[max_iou < 0.5] = 0
                roi_labels_list.append(labels)
                roi_reg_list.append(_bbox_to_delta(props, gt_b[max_idx]))

        if boxes_list:
            rois_px = lucid.concatenate(boxes_list, axis=0)
            roi_idx = lucid.concatenate(idx_list, axis=0)

            roi_labels = lucid.concatenate(roi_labels_list, axis=0)
            roi_reg_targets = lucid.concatenate(roi_reg_list, axis=0)

        else:
            rois_px = lucid.empty(0, 4, device=images.device)
            roi_idx = lucid.empty(0, dtype=lucid.Int32, device=images.device)

            roi_labels = lucid.empty(0, dtype=lucid.Int32, device=images.device)
            roi_reg_targets = lucid.empty(0, 4, device=images.device)

        rois_norm = rois_px / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
        rois_norm = rois_norm.to(images.device)

        sampled_rois, sampled_idx, sampled_labels, sampled_tgts = [], [], [], []
        for i in range(B):
            mask_i = (roi_idx == i).nonzero().squeeze(1)
            labels_i = roi_labels[mask_i]

            pos_i = (labels_i > 0).nonzero().squeeze(1)
            neg_i = (labels_i == 0).nonzero().squeeze(1)

            num_pos = min(pos_i.shape[0], 32)
            num_neg = min(neg_i.shape[0], 128 - num_pos)

            perm_pos = lucid.randperm(pos_i.shape[0])[:num_pos]
            perm_neg = lucid.randperm(neg_i.shape[0])[:num_neg]
            sel = mask_i[lucid.concatenate([pos_i[perm_pos], neg_i[perm_neg]], axis=0)]

            sampled_rois.append(rois_norm[sel])
            sampled_idx.append(roi_idx[sel])

            sampled_labels.append(roi_labels[sel])
            sampled_tgts.append(roi_reg_targets[sel])

        sb_rois = lucid.concatenate(sampled_rois, axis=0)
        sb_idx = lucid.concatenate(sampled_idx, axis=0)

        sb_labels = lucid.concatenate(sampled_labels, axis=0)
        sb_targets = lucid.concatenate(sampled_tgts, axis=0)

        cls_logits, bbox_deltas = self._roi_forward(feats, sb_rois, sb_idx)
        roi_cls_loss = F.cross_entropy(cls_logits, sb_labels, reduction="mean")

        fg = (sb_labels > 0).nonzero().squeeze(1)
        if fg.shape[0] > 0:
            preds = []
            for j, c in enumerate(sb_labels[fg]):
                start = c * 4
                preds.append(bbox_deltas[fg[j], start : start + 4])

            preds = lucid.stack(preds, axis=0)
            tgt = sb_targets[fg]
            roi_reg_loss = F.huber_loss(preds, tgt, reduction="mean")
        else:
            roi_reg_loss = lucid.zeros((), dtype=lucid.Float32)

        total_loss = rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss

        return {
            "rpn_cls_loss": rpn_cls_loss / B,
            "rpn_reg_loss": rpn_reg_loss / B,
            "roi_cls_loss": roi_cls_loss,
            "roi_reg_loss": roi_reg_loss,
            "total_loss": total_loss / B,
        }
