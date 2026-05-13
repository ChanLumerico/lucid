"""Faster R-CNN object detector (Ren et al., NeurIPS 2015).

Paper: "Faster R-CNN: Towards Real-Time Object Detection with
        Region Proposal Networks"

Key advance over Fast R-CNN
---------------------------
Fast R-CNN still relies on external proposals (selective search ~2 s/image).
Faster R-CNN introduces the Region Proposal Network (RPN), a small fully-
convolutional network that shares the backbone feature map with the detector.
The RPN slides over the feature map and predicts objectness + box deltas for
k anchors at each spatial position.  This makes the full pipeline end-to-end
trainable at ~5 fps (VGG16) or ~17 fps (ZFNet).

Architecture
------------
  Image (B, C, H, W)
    ↓  VGG16 backbone (conv1_1 … conv5_3, stride 16, no pool5)
  Feature map (B, 512, H/16, W/16)
    ├─ RPN head: 3×3 conv → cls (2k) + bbox (4k) per spatial cell
    │    Anchors at 3 scales × 3 ratios = 9 per cell
    │    → NMS → top-N proposals per image
    │
    └─ RoI Pool (7×7) on proposals
         ↓
       FC head (fc6 4096, fc7 4096)
         ↓
       cls_score (K+1)  +  bbox_pred (K×4)

Training uses alternating / approximate joint optimisation.
Loss = L_rpn_cls + L_rpn_reg + L_det_cls + L_det_reg

Faithfulness notes
------------------
* VGG16 backbone identical to Fast R-CNN (conv1_1–conv5_3, pool5 removed).
* Anchors: 9 per cell (3 scales × 3 ratios); default scales 128/256/512,
  ratios 0.5/1.0/2.0 matching paper §3.1.
* RPN anchor stride = backbone stride = 16.
* RPN NMS threshold 0.7 (proposal generation), detection NMS 0.5 (per-class).
* Smooth-L1 loss with σ=3 for RPN, σ=1 for RoI head (paper §3.1.2 / §3.2).
* RPN: positive anchor if IoU ≥ 0.7 with any GT, negative if < 0.3.
* RoI head: proposal assigned positive if IoU ≥ 0.5 with any GT.
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._utils._detection import (
    AnchorGenerator,
    batched_nms,
    box_iou,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    remove_small_boxes,
    roi_pool,
)
from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig


# ---------------------------------------------------------------------------
# Smooth-L1 (reused from Fast R-CNN; σ configurable)
# ---------------------------------------------------------------------------


def _smooth_l1(x: Tensor, sigma: float = 1.0) -> Tensor:
    sigma2 = sigma * sigma
    abs_x: Tensor = lucid.abs(x)
    cond: Tensor  = abs_x < (1.0 / sigma2)
    return lucid.where(cond, 0.5 * sigma2 * x * x, abs_x - 0.5 / sigma2)


# ---------------------------------------------------------------------------
# VGG16 backbone  (identical to Fast R-CNN backbone)
# ---------------------------------------------------------------------------


def _vgg_block(in_ch: int, out_ch: int, n: int) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for i in range(n):
        layers += [nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
                   nn.ReLU(inplace=True)]
    return layers


class _VGG16Features(nn.Module):
    """VGG16 conv1_1 … conv5_3 (pool5 omitted).  Output stride = 16."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_vgg_block(in_channels, 64, 2), nn.MaxPool2d(2, stride=2),
            *_vgg_block(64, 128, 2),        nn.MaxPool2d(2, stride=2),
            *_vgg_block(128, 256, 3),       nn.MaxPool2d(2, stride=2),
            *_vgg_block(256, 512, 3),       nn.MaxPool2d(2, stride=2),
            *_vgg_block(512, 512, 3),
        )
        self.out_channels: int = 512

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.features(x))


# ---------------------------------------------------------------------------
# RPN head
# ---------------------------------------------------------------------------


class _RPNHead(nn.Module):
    """Lightweight RPN head: 3×3 conv → objectness + bbox deltas.

    Applied to the shared backbone feature map.  For each spatial cell
    the head predicts:
      - 2 × k objectness scores (fg / bg) — we use sigmoid and return
        the foreground score
      - 4 × k bbox deltas

    where k = len(anchor_sizes) × len(anchor_ratios) = 9 by default.
    """

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        super().__init__()
        self.conv      = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)     # objectness
        self.bbox_pred  = nn.Conv2d(in_channels, num_anchors * 4, 1) # deltas

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> tuple[Tensor, Tensor]:
        t: Tensor = F.relu(cast(Tensor, self.conv(x)))
        return cast(Tensor, self.cls_logits(t)), cast(Tensor, self.bbox_pred(t))


# ---------------------------------------------------------------------------
# RPN proposal generation
# ---------------------------------------------------------------------------


def _generate_proposals(
    logits: Tensor,
    deltas: Tensor,
    anchors: Tensor,
    image_size: tuple[int, int],
    pre_nms_top_n: int,
    post_nms_top_n: int,
    nms_thresh: float,
    min_size: float,
    score_thresh: float,
    bbox_weights: tuple[float, float, float, float],
) -> tuple[Tensor, Tensor]:
    """Generate proposals for a single image from one feature level.

    Args:
        logits:  (A,) objectness logits (A = H*W*k anchors).
        deltas:  (A, 4) bbox regression deltas.
        anchors: (A, 4) xyxy anchors.

    Returns:
        (proposals, scores): (K, 4) and (K,) after NMS.
    """
    scores = F.sigmoid(logits)   # (A,)

    # Top-K before NMS
    K = min(pre_nms_top_n, int(scores.shape[0]))
    order = lucid.argsort(-scores)[:K]
    scores = scores[order]
    deltas = deltas[order]
    anchors = anchors[order]

    # Decode
    props = decode_boxes(deltas, anchors, bbox_weights)
    props = clip_boxes_to_image(props, image_size)

    # Remove tiny boxes
    keep_small = remove_small_boxes(props, min_size)
    if int(keep_small.shape[0]) == 0:
        return lucid.zeros((0, 4)), lucid.zeros((0,))
    props  = props[keep_small]
    scores = scores[keep_small]

    # Score threshold
    thr_mask: list[int] = [
        i for i in range(int(scores.shape[0]))
        if float(scores[i].item()) >= score_thresh
    ]
    if not thr_mask:
        return lucid.zeros((0, 4)), lucid.zeros((0,))
    thr_t  = lucid.tensor(thr_mask) 
    props  = props[thr_t]
    scores = scores[thr_t]

    # NMS
    keep = batched_nms(
        props, scores,
        lucid.zeros(int(scores.shape[0])),
        nms_thresh,
    )
    keep   = keep[:post_nms_top_n]
    return props[keep], scores[keep]


# ---------------------------------------------------------------------------
# RoI head  (identical to Fast R-CNN _FastRCNNHead)
# ---------------------------------------------------------------------------


class _RoIHead(nn.Module):
    def __init__(
        self, in_channels: int, roi_size: int, num_classes: int, dropout: float
    ) -> None:
        super().__init__()
        flat = in_channels * roi_size * roi_size
        self.fc6       = nn.Linear(flat, 4096)
        self.fc7       = nn.Linear(4096, 4096)
        self.drop      = nn.Dropout(p=dropout)
        self.cls_score = nn.Linear(4096, num_classes + 1)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        x = x.flatten(1)
        x = cast(Tensor, self.drop(F.relu(cast(Tensor, self.fc6(x)))))
        x = cast(Tensor, self.drop(F.relu(cast(Tensor, self.fc7(x)))))
        return cast(Tensor, self.cls_score(x)), cast(Tensor, self.bbox_pred(x))


# ---------------------------------------------------------------------------
# Faster R-CNN
# ---------------------------------------------------------------------------


class FasterRCNNForObjectDetection(PretrainedModel):
    """Faster R-CNN end-to-end object detector (Ren et al., NeurIPS 2015).

    Input contract
    --------------
    ``x``       : (B, C, H, W) image batch.
    ``targets`` : optional training list of B dicts with:
                  ``"boxes"``  — (M_i, 4) xyxy ground-truth boxes
                  ``"labels"`` — (M_i,)   integer foreground class ids

    Output contract
    ---------------
    ``ObjectDetectionOutput``:
      ``logits``    : (Σ proposals_i, num_classes+1) raw class logits.
      ``pred_boxes``: (Σ proposals_i, 4) decoded xyxy boxes.
      ``loss``      : total loss (rpn + detection) when targets provided.
    """

    config_class: ClassVar[type[FasterRCNNConfig]] = FasterRCNNConfig
    base_model_prefix: ClassVar[str] = "faster_rcnn"

    def __init__(self, config: FasterRCNNConfig) -> None:
        super().__init__(config)
        self._cfg = config

        # Backbone
        self.backbone = _VGG16Features(config.in_channels)
        C = self.backbone.out_channels

        # Anchors
        k = len(config.rpn_anchor_sizes) * len(config.rpn_anchor_ratios)
        self._num_anchors = k
        self._anchor_gen = AnchorGenerator(
            sizes=(tuple(config.rpn_anchor_sizes),),
            aspect_ratios=(tuple(config.rpn_anchor_ratios),),
        )

        # RPN head
        self.rpn_head = _RPNHead(C, k)

        # RoI head
        self.roi_head = _RoIHead(
            in_channels=C,
            roi_size=config.roi_size,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # RPN proposal generation (inference)
    # ------------------------------------------------------------------

    def _run_rpn(
        self,
        feat: Tensor,
        image_size: tuple[int, int],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Run RPN on a batched feature map and return per-image proposals.

        Args:
            feat:       (B, C, H', W') shared backbone feature map.
            image_size: (H, W) of original image (for clipping).

        Returns:
            (proposals, scores): lists of length B.
        """
        B  = int(feat.shape[0])
        fH = int(feat.shape[2])
        fW = int(feat.shape[3])
        stride = 16  # VGG16 backbone stride

        # Generate anchors for this feature level
        anchors_lvl: list[Tensor] = self._anchor_gen.forward(
            [feat], image_size, [(stride, stride)]
        )
        anchors = anchors_lvl[0]  # (fH*fW*k, 4)

        # RPN head predictions
        logits_map, deltas_map = self.rpn_head(feat)
        # logits_map: (B, k, fH, fW)
        # deltas_map: (B, 4k, fH, fW)

        proposals_all: list[Tensor] = []
        scores_all:    list[Tensor] = []
        k = self._num_anchors

        for b in range(B):
            # Flatten: (k*fH*fW,) and (k*fH*fW, 4)
            lg_b = logits_map[b].reshape(-1)           # (A,)
            dl_b = deltas_map[b].reshape(k, 4, fH, fW)
            # Rearrange to (A, 4) with anchor-major order matching anchors
            dl_b = dl_b.permute(0, 2, 3, 1).reshape(-1, 4)  # (A, 4)

            props_b, sc_b = _generate_proposals(
                lg_b, dl_b, anchors, image_size,
                pre_nms_top_n=self._cfg.rpn_pre_nms_top_n,
                post_nms_top_n=self._cfg.rpn_post_nms_top_n,
                nms_thresh=self._cfg.rpn_nms_thresh,
                min_size=self._cfg.rpn_min_size,
                score_thresh=self._cfg.rpn_score_thresh,
                bbox_weights=(1.0, 1.0, 1.0, 1.0),  # RPN uses unweighted targets
            )
            proposals_all.append(props_b)
            scores_all.append(sc_b)

        return proposals_all, scores_all

    # ------------------------------------------------------------------
    # RPN training loss
    # ------------------------------------------------------------------

    def _rpn_loss(
        self,
        feat: Tensor,
        image_size: tuple[int, int],
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        """Compute RPN classification + regression loss.

        Anchor assignment:
          IoU ≥ rpn_fg_iou_thresh → positive (label 1)
          IoU < rpn_bg_iou_thresh → negative (label 0)
          otherwise              → ignored  (label -1)

        Sampling: up to 256 anchors per image (50% positive if available).
        """
        B  = int(feat.shape[0])
        fH = int(feat.shape[2])
        fW = int(feat.shape[3])
        stride = 16

        anchors_lvl = self._anchor_gen.forward([feat], image_size, [(stride, stride)])
        anchors = anchors_lvl[0]  # (A, 4)
        A = int(anchors.shape[0])

        logits_map, deltas_map = self.rpn_head(feat)
        k = self._num_anchors

        cls_losses: list[Tensor] = []
        reg_losses: list[Tensor] = []

        for b in range(B):
            gt_boxes = targets[b]["boxes"]   # (M, 4)
            M = int(gt_boxes.shape[0])

            lg_b = logits_map[b].reshape(-1)                       # (A,)
            dl_b = deltas_map[b].reshape(k, 4, fH, fW)
            dl_b = dl_b.permute(0, 2, 3, 1).reshape(-1, 4)        # (A, 4)

            if M == 0:
                # No GT — all negatives
                neg_mask: list[int] = list(range(min(256, A)))
                neg_t = lucid.tensor(neg_mask)
                lbl_neg = lucid.zeros((len(neg_mask),))
                cls_losses.append(F.binary_cross_entropy_with_logits(
                    lg_b[neg_t], lbl_neg
                ))
                continue

            iou_mat = box_iou(anchors, gt_boxes)  # (A, M)

            # Max IoU per anchor → best GT
            best_iou_list: list[float] = []
            best_gt_list:  list[int]   = []
            for a in range(A):
                best_v = -1.0
                best_g = 0
                for m in range(M):
                    v = float(iou_mat[a, m].item())
                    if v > best_v:
                        best_v = v
                        best_g = m
                best_iou_list.append(best_v)
                best_gt_list.append(best_g)

            # Also ensure every GT has at least one positive anchor
            best_anchor_for_gt: list[int] = []
            for m in range(M):
                best_v = -1.0
                best_a = 0
                for a in range(A):
                    v = float(iou_mat[a, m].item())
                    if v > best_v:
                        best_v = v
                        best_a = a
                best_anchor_for_gt.append(best_a)

            labels: list[int] = []
            for a in range(A):
                iou_v = best_iou_list[a]
                if iou_v >= self._cfg.rpn_fg_iou_thresh:
                    labels.append(1)
                elif iou_v < self._cfg.rpn_bg_iou_thresh:
                    labels.append(0)
                else:
                    labels.append(-1)
            for anc_idx in best_anchor_for_gt:
                labels[anc_idx] = 1

            # Sample up to 256 (128 pos + 128 neg)
            pos_idx = [a for a in range(A) if labels[a] == 1]
            neg_idx = [a for a in range(A) if labels[a] == 0]
            pos_idx = pos_idx[:128]
            neg_idx = neg_idx[:max(0, 256 - len(pos_idx))]

            sampled = pos_idx + neg_idx
            if not sampled:
                continue

            samp_t  = lucid.tensor(sampled)
            lbl_samp = lucid.tensor([labels[a] for a in sampled])

            # Classification loss (binary cross-entropy objectness)
            cls_losses.append(F.binary_cross_entropy_with_logits(
                lg_b[samp_t].float(), lbl_samp.float()
            ))

            # Regression loss (positive anchors only)
            if pos_idx:
                pos_t   = lucid.tensor(pos_idx)
                gt_pos  = lucid.tensor(
                    [[float(gt_boxes[best_gt_list[a], k2].item()) for k2 in range(4)]
                     for a in pos_idx]
                )
                tgt_deltas = encode_boxes(gt_pos, anchors[pos_t], (1.0, 1.0, 1.0, 1.0))
                pred_d = dl_b[pos_t]
                reg_losses.append(_smooth_l1(pred_d - tgt_deltas, sigma=3.0).mean())

        cls_loss = lucid.cat([l.reshape(1) for l in cls_losses]).mean() if cls_losses \
            else lucid.zeros((1,))
        reg_loss = lucid.cat([l.reshape(1) for l in reg_losses]).mean() if reg_losses \
            else lucid.zeros((1,))
        return cls_loss + reg_loss

    # ------------------------------------------------------------------
    # RoI head training loss  (same as Fast R-CNN)
    # ------------------------------------------------------------------

    def _roi_loss(
        self,
        proposals: list[Tensor],
        all_logits: Tensor,
        all_deltas: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        K = self._cfg.num_classes
        N_total = int(all_deltas.shape[0])

        all_cls: list[Tensor]      = []
        all_reg_tgt: list[Tensor]  = []
        all_reg_wt:  list[Tensor]  = []

        for b, (props, tgt) in enumerate(zip(proposals, targets)):
            N_i = int(props.shape[0])
            gt_b = tgt["boxes"]    # (M, 4)
            lb_b = tgt["labels"]   # (M,)
            M    = int(gt_b.shape[0])

            if M == 0:
                all_cls.append(lucid.zeros((N_i,)))
                all_reg_tgt.append(lucid.zeros((N_i, 4)))
                all_reg_wt.append(lucid.zeros((N_i,)))
                continue

            iou_mat = box_iou(props, gt_b)  # (N_i, M)

            labels_list: list[int] = []
            best_gt:     list[int] = []
            for n in range(N_i):
                best_v = -1.0
                best_g = 0
                for m in range(M):
                    v = float(iou_mat[n, m].item())
                    if v > best_v:
                        best_v = v
                        best_g = m
                if best_v >= self._cfg.roi_fg_iou_thresh:
                    labels_list.append(int(lb_b[best_g].item()))
                elif best_v < self._cfg.roi_bg_iou_thresh:
                    labels_list.append(0)
                else:
                    labels_list.append(-1)
                best_gt.append(best_g)

            matched_boxes_data = [
                [float(gt_b[best_gt[n], k2].item()) for k2 in range(4)]
                for n in range(N_i)
            ]
            matched_boxes = lucid.tensor(matched_boxes_data)
            reg_tgt = encode_boxes(matched_boxes, props, self._cfg.bbox_reg_weights)

            lbl_t  = lucid.tensor(labels_list)
            wt_t   = lucid.tensor([1.0 if l > 0 else 0.0 for l in labels_list])

            all_cls.append(lbl_t)
            all_reg_tgt.append(reg_tgt)
            all_reg_wt.append(wt_t)

        cls_labels   = lucid.cat(all_cls,     dim=0)  # (N_total,)
        reg_targets  = lucid.cat(all_reg_tgt, dim=0)  # (N_total, 4)
        reg_weights  = lucid.cat(all_reg_wt,  dim=0)  # (N_total,)

        valid_idx: list[int] = [
            n for n in range(N_total) if int(cls_labels[n].item()) >= 0
        ]
        if not valid_idx:
            return lucid.zeros((1,))
        valid_t  = lucid.tensor(valid_idx)
        cls_loss = F.cross_entropy(all_logits[valid_t], cls_labels[valid_t])

        # Regression loss (foreground proposals only)
        deltas_3d = all_deltas.reshape(N_total, K, 4)
        reg_parts: list[Tensor] = []
        for n in range(N_total):
            if float(reg_weights[n].item()) == 0.0:
                continue
            c = max(0, min(int(cls_labels[n].item()) - 1, K - 1))
            pred_d = deltas_3d[n, c]
            tgt_d  = reg_targets[n]
            reg_parts.append(_smooth_l1(pred_d - tgt_d).mean())

        reg_loss = lucid.cat([l.reshape(1) for l in reg_parts]).mean() \
            if reg_parts else lucid.zeros((1,))

        return cls_loss + reg_loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run Faster R-CNN on a batch of images.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional training targets (list of dicts with
                     "boxes" and "labels").

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``     : (Σ proposals, num_classes+1) raw class logits.
              ``pred_boxes`` : (Σ proposals, 4) decoded xyxy boxes.
              ``loss``       : total loss when targets provided.
        """
        B  = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # 1. Shared backbone
        feat = cast(Tensor, self.backbone(x))  # (B, 512, H/16, W/16)

        # 2. RPN → proposals
        proposals, _ = self._run_rpn(feat, (iH, iW))

        # 3. RoI Pool
        roi_crops = roi_pool(
            feat, proposals,
            output_size=self._cfg.roi_size,
            spatial_scale=self._cfg.spatial_scale,
        )

        # 4. RoI head
        all_logits: Tensor
        all_deltas: Tensor
        if int(roi_crops.shape[0]) > 0:
            all_logits, all_deltas = self.roi_head(roi_crops)
        else:
            K = self._cfg.num_classes
            all_logits = lucid.zeros((0, K + 1))
            all_deltas = lucid.zeros((0, K * 4))

        # 5. Decode top-class boxes
        pred_boxes = self._decode_boxes(proposals, all_deltas, (iH, iW))

        # 6. Losses
        loss: Tensor | None = None
        if targets is not None:
            loss_rpn = self._rpn_loss(feat, (iH, iW), targets)
            loss_roi = self._roi_loss(proposals, all_logits, all_deltas, targets)
            loss     = loss_rpn + loss_roi

        return ObjectDetectionOutput(
            logits=all_logits,
            pred_boxes=pred_boxes,
            loss=loss,
        )

    def _decode_boxes(
        self,
        proposals: list[Tensor],
        all_deltas: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        if not any(int(p.shape[0]) > 0 for p in proposals):
            return lucid.zeros((0, 4))
        flat_props = lucid.cat(
            [p for p in proposals if int(p.shape[0]) > 0], dim=0
        )
        N = int(all_deltas.shape[0])
        if N == 0:
            return lucid.zeros((0, 4))
        K = self._cfg.num_classes
        # Use class-0 (first fg) delta as canonical decode
        top_deltas = all_deltas.reshape(N, K, 4)[:, 0, :]
        boxes = decode_boxes(top_deltas, flat_props, self._cfg.bbox_reg_weights)
        return clip_boxes_to_image(boxes, image_size)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        proposals: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        """Per-class NMS on raw detector output.

        Returns list of per-image result dicts with "boxes", "scores", "labels".
        """
        logits    = output.logits
        pred_boxes = output.pred_boxes
        results:  list[dict[str, Tensor]] = []
        offset = 0

        for props in proposals:
            N_i = int(props.shape[0])
            lg_i = logits[offset: offset + N_i]
            bx_i = pred_boxes[offset: offset + N_i]
            offset += N_i

            scores_i = F.softmax(lg_i, dim=-1)
            keep_boxes:  list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(1, self._cfg.num_classes + 1):
                sc_c_all = scores_i[:, c]
                mask: list[int] = [
                    i for i in range(N_i)
                    if float(sc_c_all[i].item()) >= self._cfg.score_thresh
                ]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask)
                sc_c = sc_c_all[mask_t]
                bx_c = bx_i[mask_t]
                keep = batched_nms(
                    bx_c, sc_c, lucid.zeros(int(sc_c.shape[0])),
                    self._cfg.nms_thresh,
                )
                keep = keep[:self._cfg.max_detections]
                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(lucid.full((int(keep.shape[0]),), float(c)))

            if keep_boxes:
                results.append({
                    "boxes":  lucid.cat(keep_boxes,  dim=0),
                    "scores": lucid.cat(keep_scores, dim=0),
                    "labels": lucid.cat(keep_labels, dim=0),
                })
            else:
                results.append({
                    "boxes":  lucid.zeros((0, 4)),
                    "scores": lucid.zeros((0,)),
                    "labels": lucid.zeros((0,)),
                })
        return results
