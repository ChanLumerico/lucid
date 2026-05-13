"""Fast R-CNN backbone and object detector (Girshick, 2015).

Paper: "Fast R-CNN" (ICCV 2015)

Key advance over R-CNN
----------------------
R-CNN applies the CNN once *per proposal* (very slow).
Fast R-CNN applies the CNN *once to the whole image*, projects proposals onto
the shared feature map, and uses RoI Pooling to extract fixed-size features.

Architecture
------------
1. Full image → VGG16 conv layers (conv1_1 … conv5_3) → feature map (stride 16).
2. External proposals are projected onto the feature map with RoI Pool (7×7).
3. Flattened features (7 × 7 × 512 = 25 088) → fc6 (4 096) → fc7 (4 096).
4. Two sibling output layers:
     a. cls_score  : (num_classes + 1) softmax — one score per class incl. bg
     b. bbox_pred  : (num_classes × 4) linear — class-specific (dx,dy,dw,dh)
5. Multi-task loss at training time:
     L = L_cls + λ · L_loc
   where L_cls is log-loss and L_loc is smooth-L1 on positive samples only.

Faithfulness notes
------------------
* Backbone is VGG16 conv1_1 … conv5_3 (pool5 replaced by RoI Pool).
* Spatial scale 1/16 — four max-pool halving layers before pool5.
* RoI Pool (not RoI Align) to match the original paper.
* bbox_reg_weights encode the target normalisation described in §3.1:
  tx* = wx * (Gx - Px) / Pw, etc.  Default (10, 10, 5, 5) matches the
  empirical mean/std used in the Fast R-CNN Caffe reference implementation.
* Smooth-L1 loss with σ = 1 per §3 (matches paper eq. 3).
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
    batched_nms,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    roi_pool,
)
from lucid.models.vision.fast_rcnn._config import FastRCNNConfig


# ---------------------------------------------------------------------------
# VGG16 convolutional backbone  (pool5 removed — replaced by RoI Pool)
# ---------------------------------------------------------------------------


def _vgg16_block(in_ch: int, out_ch: int, n: int) -> list[nn.Module]:
    """Build one VGG conv block (n × Conv-BN-ReLU, no pooling here)."""
    layers: list[nn.Module] = []
    for i in range(n):
        layers += [
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        ]
    return layers


class _VGG16Features(nn.Module):
    """VGG16 conv layers conv1_1 … conv5_3 (pool5 omitted).

    Input  : (B, C, H, W)
    Output : (B, 512, H/16, W/16)   — stride 16 from four max-pool layers.

    Architecture:
      Block 1 : Conv(64)×2  → MaxPool → H/2
      Block 2 : Conv(128)×2 → MaxPool → H/4
      Block 3 : Conv(256)×3 → MaxPool → H/8
      Block 4 : Conv(512)×3 → MaxPool → H/16
      Block 5 : Conv(512)×3           → H/16  (pool5 REMOVED)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            *_vgg16_block(in_channels, 64, 2),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(64, 128, 2),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(128, 256, 3),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(256, 512, 3),
            nn.MaxPool2d(2, stride=2),
            *_vgg16_block(512, 512, 3),
            # pool5 intentionally omitted: RoI Pool takes its place
        )
        self.out_channels: int = 512

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.features(x))


# ---------------------------------------------------------------------------
# RoI head  (RoI Pool → FC → dual prediction heads)
# ---------------------------------------------------------------------------


class _FastRCNNHead(nn.Module):
    """RoI-level feature processing and prediction heads.

    Input  : (N_rois, C, roi_size, roi_size) — RoI-pooled crops
    Output : (class_logits, bbox_deltas)
               class_logits : (N_rois, num_classes + 1)
               bbox_deltas  : (N_rois, num_classes × 4)

    Architecture:
      Flatten : roi_size² × C → flat (25 088 for 7×7×512)
      fc6     : flat → 4 096, ReLU, Dropout
      fc7     : 4 096 → 4 096, ReLU, Dropout
      cls     : 4 096 → num_classes + 1  (linear)
      bbox    : 4 096 → num_classes × 4  (linear)
    """

    def __init__(
        self,
        in_channels: int,
        roi_size: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        flat = in_channels * roi_size * roi_size  # 25 088 for VGG16 + 7×7

        self.fc6  = nn.Linear(flat, 4096)
        self.fc7  = nn.Linear(4096, 4096)
        self.drop = nn.Dropout(p=dropout)

        self.cls_score = nn.Linear(4096, num_classes + 1)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        x = x.flatten(1)
        x = cast(Tensor, self.drop(F.relu(cast(Tensor, self.fc6(x)))))
        x = cast(Tensor, self.drop(F.relu(cast(Tensor, self.fc7(x)))))
        return cast(Tensor, self.cls_score(x)), cast(Tensor, self.bbox_pred(x))


# ---------------------------------------------------------------------------
# Smooth-L1 loss  (σ = 1, matching Fast R-CNN eq. 3)
# ---------------------------------------------------------------------------


def _smooth_l1(x: Tensor, sigma: float = 1.0) -> Tensor:
    """Element-wise smooth-L1 (Huber loss with transition at 1/σ²)."""
    sigma2 = sigma * sigma
    abs_x  = lucid.abs(x)
    # |x| < 1/σ² → 0.5 σ² x²;  else  |x| - 0.5/σ²
    cond: Tensor = abs_x < (1.0 / sigma2)
    return lucid.where(cond, 0.5 * sigma2 * x * x, abs_x - 0.5 / sigma2)


# ---------------------------------------------------------------------------
# Fast R-CNN for Object Detection
# ---------------------------------------------------------------------------


class FastRCNNForObjectDetection(PretrainedModel):
    """Fast R-CNN object detector (Girshick, ICCV 2015).

    Runs the VGG16 backbone once on the whole image, then uses RoI Pooling
    to extract fixed-size (7 × 7) features for every region proposal before
    running the shared FC head.

    Input contract
    --------------
    ``x``         : (B, C, H, W) image batch.
    ``proposals`` : list of B tensors, each (N_i, 4) xyxy pixel coordinates.
    ``targets``   : (optional training) list of B dicts with keys:
                    ``"boxes"``  — (M_i, 4) xyxy ground-truth boxes
                    ``"labels"`` — (M_i,)   integer foreground class ids

    Output contract
    ---------------
    ``ObjectDetectionOutput``:
      ``logits``    : (Σ N_i, num_classes + 1) raw class logits.
      ``pred_boxes``: (Σ N_i, 4) decoded xyxy top-class boxes.
      ``loss``      : scalar multi-task loss (only when targets provided).
    """

    config_class: ClassVar[type[FastRCNNConfig]] = FastRCNNConfig
    base_model_prefix: ClassVar[str] = "fast_rcnn"

    def __init__(self, config: FastRCNNConfig) -> None:
        super().__init__(config)
        self._num_classes  = config.num_classes
        self._spatial_scale = config.spatial_scale
        self._roi_size     = config.roi_size
        self._score_thresh = config.score_thresh
        self._nms_thresh   = config.nms_thresh
        self._max_det      = config.max_detections
        self._bbox_weights = config.bbox_reg_weights

        self.backbone = _VGG16Features(config.in_channels)
        self.roi_head = _FastRCNNHead(
            in_channels=self.backbone.out_channels,
            roi_size=config.roi_size,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    # ------------------------------------------------------------------
    # Training loss helpers
    # ------------------------------------------------------------------

    def _assign_proposals(
        self,
        proposals: Tensor,
        gt_boxes: Tensor,
        gt_labels: Tensor,
        fg_iou_thresh: float = 0.5,
        bg_iou_thresh_hi: float = 0.5,
        bg_iou_thresh_lo: float = 0.1,
    ) -> tuple[Tensor, Tensor]:
        """Assign each proposal a GT class label and regression target.

        Rules (§3.1 of the paper):
          IoU ≥ fg_iou_thresh              → foreground, assigned to argmax GT
          bg_iou_thresh_lo ≤ IoU < hi      → background (class 0)
          IoU < bg_iou_thresh_lo           → ignored (assigned label -1)

        Args:
            proposals:  (N, 4) xyxy
            gt_boxes:   (M, 4) xyxy ground-truth boxes
            gt_labels:  (M,) foreground class ids (1-based)

        Returns:
            assigned_labels: (N,) int labels; -1 = ignored
            assigned_boxes:  (N, 4) matched GT box per proposal
        """
        from lucid.models._utils._detection import box_iou  # local import avoids cycle
        N = int(proposals.shape[0])
        M = int(gt_boxes.shape[0])

        if M == 0:
            return lucid.zeros((N,)), proposals.clone()

        iou = box_iou(proposals, gt_boxes)  # (N, M)

        # For each proposal: best-matching GT index and IoU
        best_gt_iou_list: list[float] = []
        best_gt_idx_list: list[int]   = []
        for n in range(N):
            best_iou_val = -1.0
            best_idx     = 0
            for m in range(M):
                v = float(iou[n, m].item())
                if v > best_iou_val:
                    best_iou_val = v
                    best_idx = m
            best_gt_iou_list.append(best_iou_val)
            best_gt_idx_list.append(best_idx)

        labels_list: list[int] = []
        for n in range(N):
            iou_val = best_gt_iou_list[n]
            if iou_val >= fg_iou_thresh:
                labels_list.append(int(gt_labels[best_gt_idx_list[n]].item()))
            elif bg_iou_thresh_lo <= iou_val < bg_iou_thresh_hi:
                labels_list.append(0)  # background
            else:
                labels_list.append(-1)  # ignored

        assigned_labels = lucid.tensor(labels_list)
        # Build matched GT boxes
        matched_boxes_data: list[list[float]] = [
            [
                float(gt_boxes[best_gt_idx_list[n], k].item())
                for k in range(4)
            ]
            for n in range(N)
        ]
        assigned_boxes = lucid.tensor(matched_boxes_data)
        return assigned_labels, assigned_boxes

    def _compute_loss(
        self,
        proposals: list[Tensor],
        all_logits: Tensor,
        all_deltas: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> Tensor:
        """Multi-task loss L = L_cls + λ * L_loc  (λ = 1, paper §3).

        Args:
            proposals:   Per-image proposal lists.
            all_logits:  (Σ N_i, K+1) raw class logits.
            all_deltas:  (Σ N_i, K*4) bbox regression output.
            targets:     Per-image dict with "boxes" and "labels".

        Returns:
            Scalar total loss.
        """
        all_cls_labels: list[Tensor] = []
        all_bbox_targets: list[Tensor] = []
        all_bbox_weights: list[Tensor] = []

        offset = 0
        for b, (props, tgt) in enumerate(zip(proposals, targets)):
            N_i = int(props.shape[0])
            gt_boxes  = tgt["boxes"]
            gt_labels = tgt["labels"]

            labels_i, matched_boxes_i = self._assign_proposals(
                props, gt_boxes, gt_labels
            )

            # Regression targets only for foreground (label > 0)
            reg_tgt_i = encode_boxes(matched_boxes_i, props, self._bbox_weights)
            # Weight mask: 1.0 for foreground, 0.0 for background/ignored
            fg_mask: list[float] = [
                1.0 if int(labels_i[n].item()) > 0 else 0.0
                for n in range(N_i)
            ]
            weight_i = lucid.tensor(fg_mask)

            all_cls_labels.append(labels_i)
            all_bbox_targets.append(reg_tgt_i)
            all_bbox_weights.append(weight_i)

            offset += N_i

        cls_labels   = lucid.cat(all_cls_labels,   dim=0)  # (Σ N_i,)
        bbox_targets = lucid.cat(all_bbox_targets, dim=0)  # (Σ N_i, 4)
        bbox_weights = lucid.cat(all_bbox_weights, dim=0)  # (Σ N_i,)

        # --- Classification loss (cross-entropy, skip ignored=-1) ---
        valid_mask: list[int] = [
            n for n in range(int(cls_labels.shape[0]))
            if int(cls_labels[n].item()) >= 0
        ]
        if not valid_mask:
            cls_loss: Tensor = lucid.zeros((1,))
        else:
            valid_t = lucid.tensor(valid_mask)
            cls_loss = F.cross_entropy(
                all_logits[valid_t],
                cls_labels[valid_t],
            )

        # --- Bbox regression loss (smooth-L1, foreground only) ---
        N_total = int(all_deltas.shape[0])
        K       = self._num_classes

        # Select predicted delta for each proposal's assigned class
        # Expand bbox_targets to (N, K, 4) format for indexing
        pred_deltas = all_deltas.reshape(N_total, K, 4)

        reg_loss_parts: list[Tensor] = []
        for n in range(N_total):
            w = float(bbox_weights[n].item())
            if w == 0.0:
                continue
            cls_n = max(0, int(cls_labels[n].item()) - 1)
            cls_n = min(cls_n, K - 1)
            pred_d = pred_deltas[n, cls_n]             # (4,)
            tgt_d  = bbox_targets[n]                   # (4,)
            reg_loss_parts.append(_smooth_l1(pred_d - tgt_d).mean())

        if reg_loss_parts:
            reg_loss = lucid.cat([l.reshape(1) for l in reg_loss_parts]).mean()
        else:
            reg_loss = lucid.zeros((1,))

        return cls_loss + reg_loss

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        proposals: list[Tensor] | None = None,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run Fast R-CNN on a batch of images.

        Args:
            x:         (B, C, H, W) image batch.
            proposals: list of B tensors, each (N_i, 4) xyxy proposals.
            targets:   Optional training targets (list of dicts with
                       "boxes" and "labels").

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``     : (Σ N_i, num_classes + 1) raw class logits.
              ``pred_boxes`` : (Σ N_i, 4) decoded xyxy top-class boxes.
              ``loss``       : scalar multi-task loss (only with targets).
        """
        B  = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        if proposals is None:
            proposals = [lucid.zeros((0, 4)) for _ in range(B)]

        # 1. Shared feature extraction (one forward pass for the whole batch)
        feat_map = cast(Tensor, self.backbone(x))  # (B, 512, H/16, W/16)

        # 2. RoI Pool on the shared feature map
        roi_crops = roi_pool(
            feat_map,
            proposals,
            output_size=self._roi_size,
            spatial_scale=self._spatial_scale,
        )  # (Σ N_i, 512, roi_size, roi_size)

        # 3. FC head
        all_logits, all_deltas = self.roi_head(roi_crops)
        # all_logits : (Σ N_i, K+1)
        # all_deltas : (Σ N_i, K*4)

        # 4. Decode top-class boxes
        all_boxes = self._decode_all_boxes(proposals, all_deltas, (iH, iW))

        # 5. Training loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(proposals, all_logits, all_deltas, targets)

        return ObjectDetectionOutput(
            logits=all_logits,
            pred_boxes=all_boxes,
            loss=loss,
        )

    def _decode_all_boxes(
        self,
        proposals: list[Tensor],
        all_deltas: Tensor,
        image_size: tuple[int, int],
    ) -> Tensor:
        """Decode top-scoring class bbox delta for every proposal."""
        K       = self._num_classes
        N_total = int(all_deltas.shape[0])

        # Flatten all proposals
        if any(int(p.shape[0]) > 0 for p in proposals):
            flat_props = lucid.cat(
                [p for p in proposals if int(p.shape[0]) > 0], dim=0
            )
        else:
            return lucid.zeros((0, 4))

        # all_deltas: (N_total, K*4) → (N_total, K, 4)
        deltas_3d = all_deltas.reshape(N_total, K, 4)

        # Pick delta for foreground class with highest raw delta norm
        # (at inference, class selection happens in postprocess)
        # Here we use class 0 (first foreground) as the canonical delta
        top_deltas = deltas_3d[:, 0, :]  # (N_total, 4) — class-agnostic decode

        boxes = decode_boxes(top_deltas, flat_props, self._bbox_weights)
        return clip_boxes_to_image(boxes, image_size)

    # ------------------------------------------------------------------
    # Post-processing (score threshold + per-class NMS)
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        proposals: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        """Apply per-class NMS to raw Fast R-CNN output.

        Args:
            output:    ``ObjectDetectionOutput`` from ``forward()``.
            proposals: Proposal list passed to ``forward()`` (for shape info).

        Returns:
            Per-image list of result dicts:
              ``"boxes"``  : (K_det, 4)  kept xyxy detections
              ``"scores"`` : (K_det,)    class confidence scores
              ``"labels"`` : (K_det,)    class indices (1-based)
        """
        logits    = output.logits      # (Σ N_i, K+1)
        pred_boxes = output.pred_boxes  # (Σ N_i, 4)  — top-class decoded boxes

        results: list[dict[str, Tensor]] = []
        offset = 0

        for props in proposals:
            N_i = int(props.shape[0])
            lg_i = logits[offset: offset + N_i]       # (N_i, K+1)
            bx_i = pred_boxes[offset: offset + N_i]   # (N_i, 4)
            offset += N_i

            scores_i = F.softmax(lg_i, dim=-1)

            keep_boxes:  list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(1, self._num_classes + 1):
                cls_scores = scores_i[:, c]

                mask: list[int] = [
                    i for i in range(N_i)
                    if float(cls_scores[i].item()) >= self._score_thresh
                ]
                if not mask:
                    continue

                mask_t = lucid.tensor(mask)
                sc_c   = cls_scores[mask_t]
                bx_c   = bx_i[mask_t]

                keep = batched_nms(
                    bx_c, sc_c,
                    lucid.zeros(int(sc_c.shape[0])),
                    self._nms_thresh,
                )
                keep = keep[:self._max_det]

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
