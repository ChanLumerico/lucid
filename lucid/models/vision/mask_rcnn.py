from typing import TypedDict

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

from lucid.models.utils import (
    ROIAlign,
    MultiScaleROIAlign,
    apply_deltas,
    bbox_to_delta,
    nms,
    iou,
    clip_boxes,
)

from lucid.models.vision.faster_rcnn import (
    _AnchorGenerator,
    _RegionProposalNetwork,
    _ResNetFPNBackbone,
)
from lucid.models.vision.resnet import resnet_50, resnet_101


__all__ = [
    "MaskRCNN",
    "mask_rcnn_resnet_50_fpn",
    "mask_rcnn_resnet_101_fpn",
]


class _MaskHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.conv4 = nn.Conv2d(
            hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.deconv = nn.ConvTranspose2d(
            hidden_channels, hidden_channels, kernel_size=2, stride=2
        )
        self.mask_predictor = nn.Conv2d(hidden_channels, num_classes, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))

        return self.mask_predictor(x)


class _MaskRCNNLoss(TypedDict):
    rpn_cls_loss: Tensor
    rpn_reg_loss: Tensor
    roi_cls_loss: Tensor
    roi_reg_loss: Tensor
    mask_loss: Tensor
    total_loss: Tensor


class MaskRCNN(nn.Module):
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
        mask_pool_size: tuple[int, int] = (14, 14),
        mask_hidden_channels: int = 256,
        mask_out_size: int = 28,
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
        self.mask_roipool = (
            MultiScaleROIAlign(mask_pool_size)
            if self.use_fpn
            else ROIAlign(mask_pool_size)
        )

        fc_in = feat_channels * pool_size[0] * pool_size[1]
        self.fc1 = nn.Linear(fc_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.cls_score = nn.Linear(hidden_dim, num_classes)
        self.bbox_pred = nn.Linear(hidden_dim, num_classes * 4)
        self.mask_head = _MaskHead(
            feat_channels, num_classes, hidden_channels=mask_hidden_channels
        )

        self.mask_out_size = mask_out_size
        self.mask_target_pool = ROIAlign((mask_out_size, mask_out_size))

        self.bbox_reg_means: nn.Buffer
        self.bbox_reg_stds: nn.Buffer
        self.register_buffer(
            "bbox_reg_means", Tensor([0.0, 0.0, 0.0, 0.0], dtype=lucid.Float32)
        )
        self.register_buffer(
            "bbox_reg_stds", Tensor([0.1, 0.1, 0.2, 0.2], dtype=lucid.Float32)
        )

    def _extract_features(self, images: Tensor) -> tuple[Tensor, Tensor | list[Tensor]]:
        feats = self.backbone(images)
        if self.use_fpn and isinstance(feats, (list, tuple)):
            return feats[0], feats
        return feats, feats

    def _flatten_proposals(
        self, proposals: list[Tensor], device: str
    ) -> tuple[Tensor, Tensor]:
        boxes_list: list[Tensor] = []
        idx_list: list[Tensor] = []
        for i, p in enumerate(proposals):
            if p.size == 0:
                continue

            boxes_list.append(p)
            idx_list.append(
                lucid.full((p.shape[0],), i, dtype=lucid.Int32, device=device)
            )

        if boxes_list:
            return lucid.concatenate(boxes_list, axis=0), lucid.concatenate(
                idx_list, axis=0
            )

        return (
            lucid.empty(0, 4, device=device),
            lucid.empty(0, dtype=lucid.Int32, device=device),
        )

    def roi_forward(
        self,
        feats: Tensor | list[Tensor],
        rois: Tensor,
        roi_idx: Tensor,
    ) -> tuple[Tensor, Tensor]:
        pooled = self.roipool(feats, rois, roi_idx)
        N = pooled.shape[0]

        x = pooled.reshape(N, -1)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)

        cls_logits = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)

        return cls_logits, bbox_deltas

    def mask_forward(
        self, feats: Tensor | list[Tensor], rois: Tensor, roi_idx: Tensor
    ) -> Tensor:
        pooled = self.mask_roipool(feats, rois, roi_idx)
        return self.mask_head(pooled)

    def forward(
        self,
        images: Tensor,
        rois: Tensor | None = None,
        roi_idx: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        _, H, W = images.shape[1:]
        feature_rpn, feature_roi = self._extract_features(images)

        if rois is None or roi_idx is None:
            proposals = self.rpn(feature_rpn, (H, W))
            proposals = [p.detach() for p in proposals]
            rois_px, roi_idx = self._flatten_proposals(proposals, images.device)

            rois = rois_px / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
            rois = rois.to(images.device)

        cls_logits, bbox_deltas = self.roi_forward(feature_roi, rois, roi_idx)
        mask_logits = self.mask_forward(feature_roi, rois, roi_idx)

        return cls_logits, bbox_deltas, mask_logits

    def _match_rois(
        self,
        proposals: list[Tensor],
        targets: list[dict[str, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor]:
        all_labels: list[Tensor] = []
        all_deltas: list[Tensor] = []
        all_gt_indices: list[Tensor] = []

        for i, props in enumerate(proposals):
            gt_boxes = targets[i]["boxes"].to(props.device)
            gt_labels = targets[i]["labels"].to(props.device)

            if props.size == 0:
                all_labels.append(
                    lucid.empty(0, dtype=lucid.Int32, device=props.device)
                )
                all_deltas.append(lucid.empty(0, 4, device=props.device))
                all_gt_indices.append(
                    lucid.empty(0, dtype=lucid.Int32, device=props.device)
                )
                continue

            if gt_boxes.size == 0:
                all_labels.append(
                    lucid.zeros(
                        (props.shape[0],), dtype=lucid.Int32, device=props.device
                    )
                )
                all_deltas.append(lucid.zeros((props.shape[0], 4), device=props.device))
                all_gt_indices.append(
                    lucid.zeros(
                        (props.shape[0],), dtype=lucid.Int32, device=props.device
                    )
                )
                continue

            ious = iou(props, gt_boxes)
            max_iou = lucid.max(ious, axis=1)
            argmax = lucid.argsort(ious, axis=1, descending=True)[:, 0]

            labels = gt_labels[argmax].astype(lucid.Int32)
            labels[max_iou < 0.5] = 0

            deltas = bbox_to_delta(props, gt_boxes[argmax])
            all_labels.append(labels)
            all_deltas.append(deltas)
            all_gt_indices.append(argmax.astype(lucid.Int32))

        return (
            lucid.concatenate(all_labels, axis=0),
            lucid.concatenate(all_deltas, axis=0),
            lucid.concatenate(all_gt_indices, axis=0),
        )

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
        feature_rpn, feature_roi = self._extract_features(images)

        proposals = self.rpn(feature_rpn, (H, W))
        proposals = [p.detach() for p in proposals]

        rois_px, roi_idx = self._flatten_proposals(proposals, images.device)
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

            keep = cls_scores > score_thresh
            if lucid.sum(keep) == 0:
                continue

            cls_scores = cls_scores[keep]
            boxes = boxes_all[keep]
            img_ids = roi_idx[keep]

            for img_id in img_ids.unique():
                m = img_ids == img_id
                boxes_i = boxes[m]
                scores_i = cls_scores[m]

                keep_idx = nms(boxes_i, scores_i, nms_thresh)[:top_k]
                if keep_idx.size == 0:
                    continue

                det = detections[int(img_id.item())]
                det["boxes"].append(boxes_i[keep_idx])
                det["scores"].append(scores_i[keep_idx])
                det["labels"].append(
                    lucid.full(
                        (keep_idx.size,), c, dtype=lucid.Int32, device=images.device
                    )
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

        all_boxes: list[Tensor] = []
        all_labels: list[Tensor] = []
        all_idx: list[Tensor] = []
        for i, det in enumerate(detections):
            n_det = det["boxes"].shape[0]
            if n_det == 0:
                continue
            all_boxes.append(det["boxes"])
            all_labels.append(det["labels"])
            all_idx.append(
                lucid.full((n_det,), i, dtype=lucid.Int32, device=images.device)
            )

        if all_boxes:
            det_rois_px = lucid.concatenate(all_boxes, axis=0)
            det_labels = lucid.concatenate(all_labels, axis=0)
            det_idx = lucid.concatenate(all_idx, axis=0)

            det_rois_norm = det_rois_px / lucid.Tensor(
                [W, H, W, H], dtype=lucid.Float32
            )
            det_rois_norm = det_rois_norm.to(images.device)

            mask_logits = self.mask_forward(feature_roi, det_rois_norm, det_idx)
            mask_probs: list[Tensor] = []
            for j, c in enumerate(det_labels):
                cls_idx = int(c.item())
                mask_probs.append(F.sigmoid(mask_logits[j, cls_idx]))
            mask_probs_all = lucid.stack(mask_probs, axis=0)

            offset = 0
            for det in detections:
                n_det = det["boxes"].shape[0]
                if n_det == 0:
                    det["masks"] = lucid.empty(
                        0,
                        self.mask_out_size,
                        self.mask_out_size,
                        device=images.device,
                    )
                else:
                    det["masks"] = mask_probs_all[offset : offset + n_det]
                offset += n_det

        else:
            for det in detections:
                det["masks"] = lucid.empty(
                    0,
                    self.mask_out_size,
                    self.mask_out_size,
                    device=images.device,
                )

        return detections

    def get_loss(
        self, images: Tensor, targets: list[dict[str, Tensor]]
    ) -> _MaskRCNNLoss:
        B, _, H, W = images.shape
        feature_rpn, feature_roi = self._extract_features(images)

        rpn_out = self.rpn.get_loss(feature_rpn, targets, (H, W))
        rpn_cls_loss = rpn_out["rpn_cls_loss"]
        rpn_reg_loss = rpn_out["rpn_reg_loss"]
        proposals = [p.detach() for p in rpn_out["proposals"]]

        rois_px, roi_idx = self._flatten_proposals(proposals, images.device)
        rois_norm = rois_px / lucid.Tensor([W, H, W, H], dtype=lucid.Float32)
        rois_norm = rois_norm.to(images.device)

        roi_labels, roi_reg_targets, roi_gt_indices = self._match_rois(
            proposals, targets
        )

        sampled_rois: list[Tensor] = []
        sampled_idx: list[Tensor] = []
        sampled_labels: list[Tensor] = []
        sampled_tgts: list[Tensor] = []
        sampled_gt_indices: list[Tensor] = []

        for i in range(B):
            mask_i = (roi_idx == i).nonzero().squeeze(axis=1)
            labels_i = roi_labels[mask_i]
            pos_i = (labels_i > 0).nonzero().squeeze(axis=1)
            neg_i = (labels_i == 0).nonzero().squeeze(axis=1)

            num_pos = min(pos_i.shape[0], 32)
            num_neg = min(neg_i.shape[0], 128 - num_pos)

            perm_pos = lucid.random.permutation(pos_i.shape[0], device=images.device)[
                :num_pos
            ]
            perm_neg = lucid.random.permutation(neg_i.shape[0], device=images.device)[
                :num_neg
            ]

            sel_idx = lucid.concatenate([pos_i[perm_pos], neg_i[perm_neg]], axis=0)
            sel = mask_i[sel_idx]

            sampled_rois.append(rois_norm[sel])
            sampled_idx.append(roi_idx[sel])
            sampled_labels.append(roi_labels[sel])
            sampled_tgts.append(roi_reg_targets[sel])
            sampled_gt_indices.append(roi_gt_indices[sel])

        sb_rois = lucid.concatenate(sampled_rois, axis=0)
        sb_idx = lucid.concatenate(sampled_idx, axis=0)
        sb_labels = lucid.concatenate(sampled_labels, axis=0)
        sb_tgts = lucid.concatenate(sampled_tgts, axis=0)
        sb_gt_idx = lucid.concatenate(sampled_gt_indices, axis=0)

        valid = (sb_rois[:, 2] > sb_rois[:, 0]) & (sb_rois[:, 3] > sb_rois[:, 1])
        if lucid.sum(valid) > 0:
            sb_rois = sb_rois[valid]
            sb_idx = sb_idx[valid]
            sb_labels = sb_labels[valid]
            sb_tgts = sb_tgts[valid]
            sb_gt_idx = sb_gt_idx[valid]
        else:
            sb_rois = lucid.empty(0, 4, device=images.device)

        if sb_rois.shape[0] == 0:
            zero = lucid.zeros((), dtype=lucid.Float32, device=images.device)
            total_loss = (rpn_cls_loss + rpn_reg_loss) / B
            return {
                "rpn_cls_loss": rpn_cls_loss,
                "rpn_reg_loss": rpn_reg_loss,
                "roi_cls_loss": zero,
                "roi_reg_loss": zero,
                "mask_loss": zero,
                "total_loss": total_loss,
            }

        cls_logits, bbox_deltas = self.roi_forward(feature_roi, sb_rois, sb_idx)
        roi_cls_loss = F.cross_entropy(cls_logits, sb_labels, reduction="mean")

        fg = (sb_labels > 0).nonzero().squeeze(axis=1)
        if fg.shape[0] > 0:
            preds: list[Tensor] = []
            for j, c in enumerate(sb_labels[fg]):
                start = int(c.item()) * 4
                preds.append(bbox_deltas[fg[j], start : start + 4])

            preds = lucid.stack(preds, axis=0)
            reg_tgt_fg = sb_tgts[fg]
            roi_reg_loss = F.huber_loss(preds, reg_tgt_fg, reduction="mean")

            fg_rois = sb_rois[fg]
            fg_idx = sb_idx[fg]
            fg_labels = sb_labels[fg]
            fg_gt_idx = sb_gt_idx[fg]

            mask_logits = self.mask_forward(feature_roi, fg_rois, fg_idx)
            pred_masks: list[Tensor] = []
            for j, c in enumerate(fg_labels):
                pred_masks.append(mask_logits[j, int(c.item())])
            pred_masks_t = lucid.stack(pred_masks, axis=0)

            target_masks: list[Tensor] = []
            for j in range(fg_rois.shape[0]):
                img_id = int(fg_idx[j].item())
                gt_id = int(fg_gt_idx[j].item())

                gt_masks_i = targets[img_id]["masks"].to(images.device)
                if gt_masks_i.ndim == 2:
                    gt_mask = gt_masks_i
                elif gt_masks_i.ndim == 3:
                    gt_mask = gt_masks_i[gt_id]
                elif gt_masks_i.ndim == 4:
                    gt_mask = gt_masks_i[gt_id, 0]
                else:
                    raise ValueError(
                        "targets[i]['masks'] must be (N,H,W), (N,1,H,W), or (H,W)"
                    )

                mask_img = (
                    gt_mask.astype(lucid.Float32).unsqueeze(axis=0).unsqueeze(axis=0)
                )
                roi_j = fg_rois[j].unsqueeze(axis=0)
                roi_j_idx = lucid.zeros((1,), dtype=lucid.Int32, device=images.device)

                pooled_tgt = self.mask_target_pool(mask_img, roi_j, roi_j_idx)
                if pooled_tgt.shape[0] == 0:
                    target_mask = lucid.zeros(
                        (self.mask_out_size, self.mask_out_size),
                        dtype=lucid.Float32,
                        device=images.device,
                    )
                else:
                    target_mask = pooled_tgt[0, 0]

                target_masks.append((target_mask >= 0.5).astype(lucid.Float32))

            target_masks_t = lucid.stack(target_masks, axis=0)
            mask_loss = F.binary_cross_entropy_with_logits(
                pred_masks_t, target_masks_t, reduction="mean"
            )

        else:
            roi_reg_loss = lucid.zeros((), dtype=lucid.Float32, device=images.device)
            mask_loss = lucid.zeros((), dtype=lucid.Float32, device=images.device)

        total_loss = (
            rpn_cls_loss + rpn_reg_loss + roi_cls_loss + roi_reg_loss + mask_loss
        ) / B
        return {
            "rpn_cls_loss": rpn_cls_loss,
            "rpn_reg_loss": rpn_reg_loss,
            "roi_cls_loss": roi_cls_loss,
            "roi_reg_loss": roi_reg_loss,
            "mask_loss": mask_loss,
            "total_loss": total_loss,
        }


@register_model
def mask_rcnn_resnet_50_fpn(
    num_classes: int = 21, backbone_num_classes: int = 1000, **kwargs
) -> MaskRCNN:
    backbone = resnet_50(backbone_num_classes)
    backbone = _ResNetFPNBackbone(backbone)
    return MaskRCNN(
        backbone,
        feat_channels=256,
        num_classes=num_classes,
        use_fpn=True,
        hidden_dim=1024,
        **kwargs,
    )


@register_model
def mask_rcnn_resnet_101_fpn(
    num_classes: int = 21, backbone_num_classes: int = 1000, **kwargs
) -> MaskRCNN:
    backbone = resnet_101(backbone_num_classes)
    backbone = _ResNetFPNBackbone(backbone)
    return MaskRCNN(
        backbone,
        feat_channels=256,
        num_classes=num_classes,
        use_fpn=True,
        hidden_dim=1024,
        **kwargs,
    )
