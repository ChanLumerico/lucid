import math
from typing import TypedDict

from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from lucid.types import _DeviceType

from lucid.models.objdet.util import (
    ROIAlign,
    MultiScaleROIAlign,
    FPN,
    apply_deltas,
    bbox_to_delta,
    nms,
    iou,
    clip_boxes,
)

from lucid.models.imgclf.resnet import resnet_50, resnet_101


__all__ = [
    "FasterRCNN",
    "faster_rcnn_resnet_50_fpn",
    "faster_rcnn_resnet_101_fpn",
]


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


class _RPNLoss(TypedDict):
    rpn_cls_loss: Tensor
    rpn_reg_loss: Tensor
    proposals: list[Tensor]


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

    def get_loss(
        self,
        feats: Tensor,
        targets: list[dict[str, Tensor]],
        image_shape: tuple[int, int],
    ) -> _RPNLoss:
        B, _, H, W = feats.shape
        logits, deltas = self.head(feats)
        logits = logits.transpose((0, 2, 3, 1)).reshape(B, -1)
        deltas = deltas.transpose((0, 2, 3, 1)).reshape(B, -1, 4)

        anchors = self.anchor_generator.grid_anchors(H, W, feats.device)

        rpn_cls_loss = lucid.zeros((), dtype=lucid.Float32)
        rpn_reg_loss = lucid.zeros((), dtype=lucid.Float32)
        proposals: list[Tensor] = []

        for b in range(B):
            gt = targets[b]["boxes"].to(feats.device)
            if gt.size == 0:
                labels = lucid.zeros(
                    anchors.shape[0], dtype=lucid.Int32, device=feats.device
                )
                reg_tgt = lucid.zeros_like(anchors)
            else:
                ious = iou(anchors, gt)
                max_iou = lucid.max(ious, axis=1)
                argmax = lucid.argsort(ious, axis=1, descending=True)[:, 0]

                labels = lucid.full(
                    (anchors.shape[0],), -1, dtype=lucid.Int32, device=feats.device
                )
                labels[max_iou < 0.3] = 0
                labels[max_iou >= 0.7] = 1

                for gi in range(gt.shape[0]):
                    best = lucid.argmax(ious[:, gi])
                    labels[best] = 1

                reg_tgt = lucid.zeros(anchors.shape[0], 4, device=feats.device)
                pos_mask = labels == 1
                if lucid.sum(pos_mask) > 0:
                    matched = gt[argmax[pos_mask]]
                    reg_tgt[pos_mask] = bbox_to_delta(anchors[pos_mask], matched)

            pos_idx = (labels == 1).nonzero().squeeze(axis=1)
            neg_idx = (labels == 0).nonzero().squeeze(axis=1)

            num_pos = min(pos_idx.shape[0], 128)
            num_neg = min(neg_idx.shape[0], 256 - num_pos)

            perm_pos = lucid.random.permutation(pos_idx.shape[0])[:num_pos]
            perm_neg = lucid.random.permutation(neg_idx.shape[0])[:num_neg]
            samp = lucid.concatenate([pos_idx[perm_pos], neg_idx[perm_neg]], axis=0)

            cls_scores = F.sigmoid(logits[b][samp])
            cls_targets = labels[samp].astype(lucid.Float32)
            rpn_cls_loss += F.binary_cross_entropy(
                cls_scores, cls_targets, reduction="mean"
            )

            if num_pos > 0:
                reg_pred = deltas[b][pos_idx[perm_pos]]
                reg_true = reg_tgt[pos_idx[perm_pos]]
                rpn_reg_loss += F.huber_loss(reg_pred, reg_true, reduction="mean")

            sc = F.sigmoid(logits[b])
            bx = apply_deltas(anchors, deltas[b], add_one=1.0)
            bx = clip_boxes(bx, image_shape)

            keep = sc > self.score_thresh
            if lucid.sum(keep) == 0:
                proposals.append(lucid.empty(0, 4, device=feats.device))
            else:
                sc_k, bx_k = sc[keep], bx[keep]
                order = lucid.argsort(sc_k, descending=True)[: self.pre_nms_top_n]
                bx_k, sc_k = bx_k[order], sc_k[order]

                keep_idx = nms(bx_k, sc_k, self.nms_thresh)[: self.post_nms_top_n]
                proposals.append(bx_k[keep_idx])

        rpn_cls_loss /= B
        rpn_reg_loss /= B

        return {
            "rpn_cls_loss": rpn_cls_loss,
            "rpn_reg_loss": rpn_reg_loss,
            "proposals": proposals,
        }


class _FasterRCNNLoss(TypedDict):
    rpn_cls_loss: Tensor
    rpn_reg_loss: Tensor
    roi_cls_loss: Tensor
    roi_reg_loss: Tensor
    total_loss: Tensor


class FasterRCNN(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feat_channels: int,
        num_classes: int,
        *,
        use_fpn: bool = False,
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
        self.use_fpn = use_fpn
        self.roipool = (
            MultiScaleROIAlign(pool_size) if self.use_fpn else ROIAlign(pool_size)
        )

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

    def roi_forward(
        self,
        feats: Tensor | list[Tensor],
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
        if self.use_fpn and isinstance(feats, (list, tuple)):
            feature_rpn = feats[0]
            feature_roi = feats
        else:
            feature_rpn = feats
            feature_roi = feats

        if rois is None or roi_idx is None:
            proposals = self.rpn(feature_rpn, (H, W))
            proposals = [p.detach() for p in proposals]

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

        return self.roi_forward(feature_roi, rois, roi_idx, return_feats=return_feats)

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
        if self.use_fpn and isinstance(feats, (list, tuple)):
            feature_rpn = feats[0]
            feature_roi = feats
        else:
            feature_rpn = feats
            feature_roi = feats

        proposals = self.rpn(feature_rpn, (H, W))
        proposals = [p.detach() for p in proposals]

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

        cls_logits, bbox_deltas = self.roi_forward(feature_roi, rois_norm, roi_idx)
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

    def _match_rois(
        self, proposals: list[Tensor], targets: list[dict[str, Tensor]]
    ) -> tuple[Tensor, Tensor]:
        all_labels: list[Tensor] = []
        all_deltas: list[Tensor] = []

        for i, props in enumerate(proposals):
            gt_boxes = targets[i]["boxes"].to(props.device)
            gt_labels = targets[i]["labels"].to(props.device)

            if props.size == 0:
                all_labels.append(
                    lucid.empty(0, dtype=lucid.Int32, device=props.device)
                )
                all_deltas.append(lucid.empty(0, 4, device=props.device))
                continue

            ious = iou(props, gt_boxes)
            max_iou = lucid.max(ious, axis=1)
            argmax = lucid.argsort(ious, axis=1, descending=True)[:, 0]

            labels = gt_labels[argmax].astype(lucid.Int32)
            labels[max_iou < 0.5] = 0

            deltas = bbox_to_delta(props, gt_labels[argmax])
            all_labels.append(labels)
            all_deltas.append(deltas)

        roi_labels = lucid.concatenate(all_labels, axis=0)
        roi_reg_tgts = lucid.concatenate(all_deltas, axis=0)

        return roi_labels, roi_reg_tgts

    def get_loss(
        self, images: Tensor, targets: list[dict[str, Tensor]]
    ) -> _FasterRCNNLoss:
        B, _, H, W = images.shape
        feats = self.backbone(images)
        if self.use_fpn and isinstance(feats, (list, tuple)):
            feature_rpn = feats[0]
            feature_roi = feats
        else:
            feature_rpn = feats
            feature_roi = feats

        rpn_out = self.rpn.get_loss(feature_rpn, targets, (H, W))
        rpn_cls_loss = rpn_out["rpn_cls_loss"]
        rpn_reg_loss = rpn_out["rpn_reg_loss"]
        proposals = rpn_out["proposals"]
        proposals = [p.detach() for p in proposals]

        boxes_list, idx_list = [], []
        for i, props in enumerate(proposals):
            if props.size == 0:
                continue
            boxes_list.append(props)
            idx_list.append(
                lucid.full(
                    (props.shape[0],), i, dtype=lucid.Int32, device=images.device
                )
            )

        if boxes_list:
            rois_px = lucid.concatenate(boxes_list, axis=0)
            roi_idx = lucid.concatenate(idx_list, axis=0)
        else:
            rois_px = lucid.empty(0, 4, device=images.device)
            roi_idx = lucid.empty(0, dtype=lucid.Int32, device=images.device)

        rois_norm = rois_px / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
        rois_norm = rois_norm.to(images.device)

        sampled_rois, sampled_idx, sampled_labels, sampled_tgts = [], [], [], []
        roi_labels, roi_reg_targets = self._match_rois(proposals, targets)

        for i in range(B):
            mask_i = (roi_idx == i).nonzero().squeeze(axis=1)
            labels_i = roi_labels[mask_i]
            pos_i = (labels_i > 0).nonzero().squeeze(axis=1)
            neg_i = (labels_i == 0).nonzero().squeeze(axis=1)

            num_pos = min(pos_i.shape[0], 32)
            num_neg = min(neg_i.shape[0], 128 - num_pos)

            perm_pos = lucid.random.permutation(pos_i.shape[0])[:num_pos]
            perm_neg = lucid.random.permutation(neg_i.shape[0])[:num_neg]

            sel_idx = lucid.concatenate([pos_i[perm_pos], neg_i[perm_neg]], axis=0)
            sel = mask_i[sel_idx]

            sampled_rois.append(rois_norm[sel])
            sampled_idx.append(roi_idx[sel])
            sampled_labels.append(roi_labels[sel])
            sampled_tgts.append(roi_reg_targets[sel])

        sb_rois = lucid.concatenate(sampled_rois, axis=0)
        sb_idx = lucid.concatenate(sampled_idx, axis=0)
        sb_labels = lucid.concatenate(sampled_labels, axis=0)
        sb_tgts = lucid.concatenate(sampled_tgts, axis=0)

        cls_logits, bbox_deltas = self.roi_forward(feature_roi, sb_rois, sb_idx)
        roi_cls_loss = F.cross_entropy(cls_logits, sb_labels, reduction="mean")

        fg = (sb_labels > 0).nonzero().squeeze(axis=1)
        if fg.shape[0] > 0:
            preds = []
            for j, c in enumerate(sb_labels[fg]):
                start = c * 4
                preds.append(bbox_deltas[fg[j], start : start + 4])

            preds = lucid.stack(preds, axis=0)
            reg_tgt_fg = sb_tgts[fg]
            roi_reg_loss = F.huber_loss(preds, reg_tgt_fg, reduction="mean")

        else:
            roi_reg_loss = lucid.zeros((), dtype=lucid.Float32)

        total_loss = (rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss) / B
        return {
            "rpn_cls_loss": rpn_cls_loss,
            "rpn_reg_loss": rpn_reg_loss,
            "roi_cls_loss": roi_cls_loss,
            "roi_reg_loss": roi_reg_loss,
            "total_loss": total_loss,
        }


class _ResNetFPNBackbone(nn.Module):
    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.stem = self.net.stem
        self.maxpool = self.net.maxpool
        self.layers = [
            self.net.layer1,
            self.net.layer2,
            self.net.layer3,
            self.net.layer4,
        ]
        self.fpn = FPN([256, 512, 1024, 2048], out_channels=256)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)
        cs = []
        for i in range(4):
            x = self.layers[i](x)
            cs.append(x)
        return self.fpn(cs)


@register_model
def faster_rcnn_resnet_50_fpn(
    num_classes: int = 21, backbone_num_classes: int = 1000, **kwargs
) -> FasterRCNN:
    backbone = resnet_50(backbone_num_classes)
    backbone = _ResNetFPNBackbone(backbone)
    return FasterRCNN(
        backbone,
        feat_channels=256,
        num_classes=num_classes,
        use_fpn=True,
        hidden_dim=1024,
        **kwargs,
    )


@register_model
def faster_rcnn_resnet_101_fpn(
    num_classes: int = 21, backbone_num_classes: int = 1000, **kwargs
) -> FasterRCNN:
    backbone = resnet_101(backbone_num_classes)
    backbone = _ResNetFPNBackbone(backbone)
    return FasterRCNN(
        backbone,
        feat_channels=256,
        num_classes=num_classes,
        use_fpn=True,
        hidden_dim=1024,
        **kwargs,
    )
