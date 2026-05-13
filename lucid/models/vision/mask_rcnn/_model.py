"""Mask R-CNN instance segmentation model (He et al., ICCV 2017).

Paper: "Mask R-CNN"

Key advances over Faster R-CNN
-------------------------------
1. **FPN backbone** — replaces the single-scale VGG16 with a ResNet-50 Feature
   Pyramid Network (P2–P5, stride 4/8/16/32; P6 added by max-pool for RPN).
2. **RoI Align** — sub-pixel bilinear sampling replaces quantised RoI Pool,
   eliminating the two-quantisation misalignment and improving mask accuracy.
3. **Mask branch** — a parallel FCN head predicts K binary masks (one per
   class) for each proposal, trained with per-pixel binary cross-entropy on
   the ground-truth class mask only.

Architecture
------------
  Image (B, C, H, W)
    ↓  ResNet-50 (conv1, pool, layer1–layer4) → C2, C3, C4, C5
    ↓  FPN (5-level: P2–P5 from backbone, P6 = max-pool of P5 for RPN)
  [P2, P3, P4, P5, P6] ─── RPN ──→ proposals (per image)
  [P2, P3, P4, P5] + proposals
      ├─ FPN-level assignment (eq. 1 in §4 of the paper)
      ├─ RoI Align (7×7) → 2-FC head → cls_score (K+1)  + bbox_pred (K×4)
      └─ RoI Align (14×14) → Mask FCN (4×conv + deconv + 1×1) → masks (K, 28, 28)

Losses (training)
-----------------
  L = L_rpn_cls + L_rpn_reg + L_det_cls + L_det_reg + L_mask

  L_rpn_*  : binary cross-entropy objectness + smooth-L1(σ=3).
  L_det_*  : cross-entropy + smooth-L1(σ=1), same as Fast/Faster R-CNN.
  L_mask   : binary cross-entropy on each foreground RoI's predicted mask
             for the GT class only (mask head uses sigmoid, not softmax).

Faithfulness notes
------------------
* ResNet-50 bottleneck blocks with batch-norm and 4× channel expansion.
* FPN follows Lin et al. (2017): lateral 1×1 + top-down nearest-neighbour
  upsample + 3×3 anti-aliasing conv; P6 by 2×2 max-pool of P5 (for RPN).
* Anchor sizes (32,64,128,256,512) per level, ratios (0.5,1.0,2.0) → 3 per cell.
* FPN-level assignment: k = k0 + ⌊log2(√(wh)/224)⌋, k0=4, k∈[2,5].
* RoI Align spatial scale = 1/stride for each FPN level.
* Mask output: 28×28 (14→28 via deconv, stride 2).
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import InstanceSegmentationOutput
from lucid.models._utils._detection import (
    AnchorGenerator,
    FPN,
    RPN,
    RoIHead,
    batched_nms,
    box_iou,
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
    roi_align,
)
from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _smooth_l1(x: Tensor, sigma: float = 1.0) -> Tensor:
    sigma2 = sigma * sigma
    abs_x: Tensor = lucid.abs(x)
    cond: Tensor = abs_x < (1.0 / sigma2)
    return lucid.where(cond, 0.5 * sigma2 * x * x, abs_x - 0.5 / sigma2)


# ---------------------------------------------------------------------------
# ResNet-50 building blocks
# ---------------------------------------------------------------------------


class _Bottleneck(nn.Module):
    """ResNet Bottleneck: 1×1 → 3×3 → 1×1 with 4× channel expansion."""

    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion

        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x

        out: Tensor = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        out = F.relu(cast(Tensor, self.bn2(cast(Tensor, self.conv2(out)))))
        out = cast(Tensor, self.bn3(cast(Tensor, self.conv3(out))))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        return F.relu(out + identity)


def _make_layer(
    in_ch: int,
    mid_ch: int,
    num_blocks: int,
    stride: int = 1,
) -> tuple[nn.Sequential, int]:
    """Build one ResNet stage; returns (layer, out_channels)."""
    out_ch = mid_ch * _Bottleneck.expansion
    downsample: nn.Module | None = None
    if stride != 1 or in_ch != out_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    blocks: list[nn.Module] = [
        _Bottleneck(in_ch, mid_ch, stride=stride, downsample=downsample)
    ]
    for _ in range(1, num_blocks):
        blocks.append(_Bottleneck(out_ch, mid_ch))

    return nn.Sequential(*blocks), out_ch


class _ResNet50Backbone(nn.Module):
    """ResNet-50 stem + four stages.

    Returns C2, C3, C4, C5 feature maps (strides 4, 8, 16, 32).
    No average pool or fully-connected head.
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        # Stem: conv1 7×7 stride-2 → BN → ReLU → MaxPool 3×3 stride-2 → stride 4
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stages (layer1 has stride 1; layers 2–4 double the spatial stride)
        self.layer1, c2 = _make_layer(64, 64, layers[0], stride=1)  # C2: 256ch
        self.layer2, c3 = _make_layer(c2, 128, layers[1], stride=2)  # C3: 512ch
        self.layer3, c4 = _make_layer(c3, 256, layers[2], stride=2)  # C4: 1024ch
        self.layer4, c5 = _make_layer(c4, 512, layers[3], stride=2)  # C5: 2048ch

        self.out_channels: list[int] = [c2, c3, c4, c5]

    def forward(self, x: Tensor) -> list[Tensor]:  # type: ignore[override]
        x = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        x = cast(Tensor, self.pool(x))

        c2: Tensor = cast(Tensor, self.layer1(x))
        c3: Tensor = cast(Tensor, self.layer2(c2))
        c4: Tensor = cast(Tensor, self.layer3(c3))
        c5: Tensor = cast(Tensor, self.layer4(c4))
        return [c2, c3, c4, c5]


# ---------------------------------------------------------------------------
# Mask head
# ---------------------------------------------------------------------------


class _MaskHead(nn.Module):
    """Mask FCN head (§3.1 of the paper).

    Architecture:
      4 × Conv(in_ch, hidden, 3, padding=1) → ReLU
      ConvTranspose2d(hidden, hidden, 2, stride=2)   [14 → 28]
      Conv(hidden, num_classes, 1)                   [logit per pixel per class]

    Output sigmoid is applied only at inference.  During training the
    binary cross-entropy loss is computed against the raw logits.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        convs: list[nn.Module] = []
        ch_in = in_channels
        for _ in range(4):
            convs.append(nn.Conv2d(ch_in, hidden_channels, 3, padding=1))
            convs.append(nn.ReLU(inplace=True))
            ch_in = hidden_channels
        self.fcn = nn.Sequential(*convs)
        self.deconv = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, stride=2)
        self.predictor = nn.Conv2d(hidden_channels, num_classes, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.fcn(x))
        x = F.relu(cast(Tensor, self.deconv(x)))
        return cast(Tensor, self.predictor(x))


# ---------------------------------------------------------------------------
# FPN-level assignment
# ---------------------------------------------------------------------------


def _assign_fpn_levels(
    proposals: list[Tensor],
    k0: int = 4,
    num_levels: int = 4,
) -> list[list[int]]:
    """Assign each proposal to an FPN level (P2–P5 → indices 0–3).

    Equation from §4 of Mask R-CNN:
        k = ⌊k0 + log2(√(w·h) / 224)⌋
        k clamped to [2, 5] → index = k − 2 in [0, 3].
    """
    per_image: list[list[int]] = []
    for props in proposals:
        levels: list[int] = []
        for n in range(int(props.shape[0])):
            w = float(props[n, 2].item()) - float(props[n, 0].item())
            h = float(props[n, 3].item()) - float(props[n, 1].item())
            scale = math.sqrt(max(w * h, 1e-6))
            k = int(k0 + math.log2(scale / 224.0 + 1e-6))
            k = max(2, min(5, k))
            levels.append(min(k - 2, num_levels - 1))
        per_image.append(levels)
    return per_image


# ---------------------------------------------------------------------------
# Gather RoI Align crops from the correct FPN level
# ---------------------------------------------------------------------------


def _fpn_roi_align(
    fpn_feats: list[Tensor],
    proposals: list[Tensor],
    level_assignments: list[list[int]],
    output_size: int,
    spatial_scales: list[float],
) -> Tensor:
    """Apply RoI Align to each proposal using its assigned FPN level.

    Returns:
        (Σ N_i, C, output_size, output_size) stacked crops, in proposal order.
    """
    num_levels = len(fpn_feats)
    C = int(fpn_feats[0].shape[1])
    B = len(proposals)

    # Collect results preserving original (b, n) order
    total_n = sum(int(p.shape[0]) for p in proposals)
    if total_n == 0:
        return lucid.zeros((0, C, output_size, output_size))

    # Build a flat list of (b, n, level) tuples in proposal order
    order: list[tuple[int, int, int]] = []
    for b in range(B):
        lvls = level_assignments[b]
        for n, lvl in enumerate(lvls):
            order.append((b, n, lvl))

    # Group by (b, level) to call roi_align per batch × level
    result_map: dict[tuple[int, int], Tensor] = {}

    # Group proposals by (b, level)
    groups: dict[tuple[int, int], list[tuple[int, Tensor]]] = {}
    for b in range(B):
        for n, lvl in enumerate(level_assignments[b]):
            key = (b, lvl)
            if key not in groups:
                groups[key] = []
            groups[key].append((n, proposals[b][n : n + 1]))  # (1, 4)

    for (b, lvl), items in groups.items():
        feat_b = fpn_feats[lvl]  # (B, C, H, W)
        scale = spatial_scales[lvl]
        boxes = lucid.cat([box for _, box in items], dim=0)  # (k, 4)
        # roi_align expects a list-per-image
        crops = roi_align(
            feat_b,
            [lucid.zeros((0, 4))] * b + [boxes] + [lucid.zeros((0, 4))] * (B - b - 1),
            output_size=output_size,
            spatial_scale=scale,
        )
        result_map[(b, lvl)] = crops

    # Reassemble in original proposal order
    crops_ordered: list[Tensor] = []
    # track per-(b,lvl) row offset
    offsets: dict[tuple[int, int], int] = {k: 0 for k in result_map}

    for b, n, lvl in order:
        key = (b, lvl)
        idx = offsets[key]
        crops_ordered.append(result_map[key][idx : idx + 1])
        offsets[key] = idx + 1

    return lucid.cat(crops_ordered, dim=0)


# ---------------------------------------------------------------------------
# Mask R-CNN
# ---------------------------------------------------------------------------


class MaskRCNNForObjectDetection(PretrainedModel):
    """Mask R-CNN end-to-end instance segmentation model (He et al., ICCV 2017).

    Input contract
    --------------
    ``x``       : (B, C, H, W) image batch.
    ``targets`` : optional training list of B dicts with:
                    ``"boxes"``  — (M_i, 4) xyxy ground-truth boxes
                    ``"labels"`` — (M_i,)   integer foreground class ids
                    ``"masks"``  — (M_i, H, W) binary instance masks

    Output contract
    ---------------
    ``InstanceSegmentationOutput``:
      ``logits``     : (Σ proposals, K+1)      raw class logits.
      ``pred_boxes`` : (Σ proposals, 4)         decoded xyxy boxes.
      ``pred_masks`` : (Σ proposals, K, 28, 28) raw mask logits.
      ``loss``       : total loss when targets provided.
    """

    config_class: ClassVar[type[MaskRCNNConfig]] = MaskRCNNConfig
    base_model_prefix: ClassVar[str] = "mask_rcnn"

    def __init__(self, config: MaskRCNNConfig) -> None:
        super().__init__(config)
        self._cfg = config
        K = config.num_classes
        C = config.fpn_out_channels

        # 1. Backbone (ResNet-50)
        self.backbone = _ResNet50Backbone(config.in_channels, config.backbone_layers)

        # 2. FPN (P2–P5 from C2–C5; extra_blocks=1 adds P6 for RPN)
        self.fpn = FPN(
            in_channels=self.backbone.out_channels,
            out_channels=C,
            extra_blocks=1,
        )

        # FPN spatial scales for detection levels P2–P5
        # P2 stride=4, P3=8, P4=16, P5=32
        self._det_spatial_scales: list[float] = [
            1.0 / 4.0,
            1.0 / 8.0,
            1.0 / 16.0,
            1.0 / 32.0,
        ]

        # 3. Anchor generator (5 levels: P2–P6)
        num_rpn_levels = len(config.rpn_anchor_sizes)
        ratios = tuple(config.rpn_anchor_ratios)
        self._anchor_gen = AnchorGenerator(
            sizes=tuple((s,) for s in config.rpn_anchor_sizes),
            aspect_ratios=(ratios,) * num_rpn_levels,
        )

        # 4. RPN
        num_anchors = len(config.rpn_anchor_ratios)  # 3 per cell per level
        self._rpn = RPN(
            in_channels=C,
            num_anchors=num_anchors,
            pre_nms_top_n=config.rpn_pre_nms_top_n,
            post_nms_top_n=config.rpn_post_nms_top_n,
            nms_threshold=config.rpn_nms_thresh,
            min_size=config.rpn_min_size,
            score_thresh=config.rpn_score_thresh,
        )

        # 5. Detection head
        self.det_head = RoIHead(
            in_channels=C,
            roi_size=config.roi_det_size,
            num_classes=K,
            representation_size=config.roi_representation,
        )

        # 6. Mask head
        self.mask_head = _MaskHead(
            in_channels=C,
            hidden_channels=config.mask_hidden_channels,
            num_classes=K,
        )

    # ------------------------------------------------------------------
    # RPN training loss
    # ------------------------------------------------------------------

    def _rpn_loss(
        self,
        fpn_feats: list[Tensor],
        image_size: tuple[int, int],
        targets: list[dict[str, Tensor]],
        strides: list[tuple[int, int]],
        anchors: list[Tensor],
    ) -> Tensor:
        """Binary cross-entropy objectness + smooth-L1 regression."""
        B = len(targets)
        cls_losses: list[Tensor] = []
        reg_losses: list[Tensor] = []

        # RPN predictions per level
        level_preds: list[tuple[Tensor, Tensor]] = []
        for feat in fpn_feats:
            t: Tensor = F.relu(cast(Tensor, self._rpn.conv(feat)))
            lg: Tensor = cast(Tensor, self._rpn.cls_logits(t))
            dl: Tensor = cast(Tensor, self._rpn.bbox_pred(t))
            level_preds.append((lg, dl))

        for b in range(B):
            gt_boxes = targets[b]["boxes"]  # (M, 4)
            M = int(gt_boxes.shape[0])

            # Flatten anchors and preds across all levels
            all_anc_parts: list[Tensor] = []
            all_lg_parts: list[Tensor] = []
            all_dl_parts: list[Tensor] = []

            for lvl_idx, (lg_map, dl_map) in enumerate(level_preds):
                A = int(lg_map.shape[1])
                fH = int(lg_map.shape[2])
                fW = int(lg_map.shape[3])
                # Spatial-major to match AnchorGenerator ordering (G*A, 4)
                lg_b = lg_map[b].permute(1, 2, 0).reshape(-1)
                dl_b = (
                    dl_map[b].reshape(A, 4, fH, fW).permute(2, 3, 0, 1).reshape(-1, 4)
                )
                all_anc_parts.append(anchors[lvl_idx])
                all_lg_parts.append(lg_b)
                all_dl_parts.append(dl_b)

            all_anc = lucid.cat(all_anc_parts, dim=0)  # (A_total, 4)
            all_lg = lucid.cat(all_lg_parts, dim=0)  # (A_total,)
            all_dl = lucid.cat(all_dl_parts, dim=0)  # (A_total, 4)
            A_total = int(all_anc.shape[0])

            if M == 0:
                neg_mask: list[int] = list(range(min(256, A_total)))
                neg_t = lucid.tensor(neg_mask).long()
                lbl_neg = lucid.zeros((len(neg_mask),))
                cls_losses.append(
                    F.binary_cross_entropy_with_logits(all_lg[neg_t], lbl_neg)
                )
                continue

            iou_mat = box_iou(all_anc, gt_boxes)  # (A_total, M)

            best_iou_list: list[float] = []
            best_gt_list: list[int] = []
            for a in range(A_total):
                best_v = -1.0
                best_g = 0
                for m in range(M):
                    v = float(iou_mat[a, m].item())
                    if v > best_v:
                        best_v = v
                        best_g = m
                best_iou_list.append(best_v)
                best_gt_list.append(best_g)

            best_anchor_for_gt: list[int] = []
            for m in range(M):
                best_v = -1.0
                best_a = 0
                for a in range(A_total):
                    v = float(iou_mat[a, m].item())
                    if v > best_v:
                        best_v = v
                        best_a = a
                best_anchor_for_gt.append(best_a)

            labels: list[int] = []
            for a in range(A_total):
                iou_v = best_iou_list[a]
                if iou_v >= self._cfg.rpn_fg_iou_thresh:
                    labels.append(1)
                elif iou_v < self._cfg.rpn_bg_iou_thresh:
                    labels.append(0)
                else:
                    labels.append(-1)
            for anc_idx in best_anchor_for_gt:
                labels[anc_idx] = 1

            pos_idx = [a for a in range(A_total) if labels[a] == 1][:128]
            neg_idx = [a for a in range(A_total) if labels[a] == 0][
                : max(0, 256 - len(pos_idx))
            ]
            sampled = pos_idx + neg_idx
            if not sampled:
                continue

            samp_t = lucid.tensor(sampled).long()
            lbl_samp = lucid.tensor([labels[a] for a in sampled])
            cls_losses.append(
                F.binary_cross_entropy_with_logits(
                    all_lg[samp_t].float(), lbl_samp.float()
                )
            )

            if pos_idx:
                pos_t = lucid.tensor(pos_idx).long()
                gt_pos = lucid.tensor(
                    [
                        [float(gt_boxes[best_gt_list[a], k2].item()) for k2 in range(4)]
                        for a in pos_idx
                    ]
                )
                tgt_d = encode_boxes(gt_pos, all_anc[pos_t], (1.0, 1.0, 1.0, 1.0))
                reg_losses.append(_smooth_l1(all_dl[pos_t] - tgt_d, sigma=3.0).mean())

        cls_l = (
            lucid.cat([l.reshape(1) for l in cls_losses]).mean()
            if cls_losses
            else lucid.zeros((1,))
        )
        reg_l = (
            lucid.cat([l.reshape(1) for l in reg_losses]).mean()
            if reg_losses
            else lucid.zeros((1,))
        )
        return cls_l + reg_l

    # ------------------------------------------------------------------
    # Detection head loss
    # ------------------------------------------------------------------

    def _det_loss(
        self,
        proposals: list[Tensor],
        all_logits: Tensor,
        all_deltas: Tensor,
        targets: list[dict[str, Tensor]],
    ) -> tuple[Tensor, list[int]]:
        """Cross-entropy + smooth-L1.  Also returns fg proposal indices."""
        K = self._cfg.num_classes
        N_total = int(all_deltas.shape[0])
        all_cls: list[Tensor] = []
        all_reg_tgt: list[Tensor] = []
        all_reg_wt: list[Tensor] = []
        fg_flags: list[int] = []  # 1 = fg, 0 = bg, flat over all proposals

        for props, tgt in zip(proposals, targets):
            N_i = int(props.shape[0])
            gt_b = tgt["boxes"]
            lb_b = tgt["labels"]
            M = int(gt_b.shape[0])

            if M == 0:
                all_cls.append(lucid.zeros((N_i,)))
                all_reg_tgt.append(lucid.zeros((N_i, 4)))
                all_reg_wt.append(lucid.zeros((N_i,)))
                fg_flags.extend([0] * N_i)
                continue

            iou_mat = box_iou(props, gt_b)
            labels_list: list[int] = []
            best_gt: list[int] = []

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

            matched_data = [
                [float(gt_b[best_gt[n], k2].item()) for k2 in range(4)]
                for n in range(N_i)
            ]
            matched_boxes = lucid.tensor(matched_data)
            reg_tgt = encode_boxes(matched_boxes, props, self._cfg.bbox_reg_weights)
            lbl_t = lucid.tensor(labels_list)
            wt_t = lucid.tensor([1.0 if l > 0 else 0.0 for l in labels_list])

            all_cls.append(lbl_t)
            all_reg_tgt.append(reg_tgt)
            all_reg_wt.append(wt_t)
            fg_flags.extend([1 if l > 0 else 0 for l in labels_list])

        cls_labels = lucid.cat(all_cls, dim=0)
        reg_targets = lucid.cat(all_reg_tgt, dim=0)
        reg_weights = lucid.cat(all_reg_wt, dim=0)

        valid_idx: list[int] = [
            n for n in range(N_total) if int(cls_labels[n].item()) >= 0
        ]
        if not valid_idx:
            return lucid.zeros((1,)), fg_flags

        valid_t = lucid.tensor(valid_idx).long()
        cls_loss = F.cross_entropy(all_logits[valid_t], cls_labels[valid_t])

        deltas_3d = all_deltas.reshape(N_total, K, 4)
        reg_parts: list[Tensor] = []
        for n in range(N_total):
            if float(reg_weights[n].item()) == 0.0:
                continue
            c = max(0, min(int(cls_labels[n].item()) - 1, K - 1))
            pred_d = deltas_3d[n, c]
            tgt_d = reg_targets[n]
            reg_parts.append(_smooth_l1(pred_d - tgt_d).mean())

        reg_loss = (
            lucid.cat([l.reshape(1) for l in reg_parts]).mean()
            if reg_parts
            else lucid.zeros((1,))
        )

        return cls_loss + reg_loss, fg_flags

    # ------------------------------------------------------------------
    # Mask head loss
    # ------------------------------------------------------------------

    def _mask_loss(
        self,
        mask_crops: Tensor,
        proposals: list[Tensor],
        targets: list[dict[str, Tensor]],
        fg_flags: list[int],
        level_assignments: list[list[int]],
    ) -> Tensor:
        """Binary cross-entropy on fg proposals' predicted masks.

        For each foreground proposal the loss is computed only for the
        GT class channel (i.e. the correct binary mask, not all K).
        """
        if int(mask_crops.shape[0]) == 0:
            return lucid.zeros((1,))

        K = self._cfg.num_classes
        mH = int(mask_crops.shape[2])  # 28
        mW = int(mask_crops.shape[3])

        mask_logits = cast(Tensor, self.mask_head(mask_crops))  # (N, K, mH, mW)

        all_losses: list[Tensor] = []
        proposal_idx = 0  # flat index into proposals / fg_flags

        for b, (props, tgt) in enumerate(zip(proposals, targets)):
            N_i = int(props.shape[0])
            gt_b = tgt["boxes"]
            lb_b = tgt["labels"]
            M = int(gt_b.shape[0])

            gt_masks_tgt = tgt.get("masks")  # (M, H_img, W_img) or None

            for n in range(N_i):
                flat_idx = proposal_idx + n
                if fg_flags[flat_idx] == 0 or M == 0:
                    continue

                # Find the best matching GT for this proposal
                best_v = -1.0
                best_g = 0
                for m in range(M):
                    iou_v = float(
                        box_iou(props[n : n + 1], gt_b[m : m + 1])[0, 0].item()
                    )
                    if iou_v > best_v:
                        best_v = iou_v
                        best_g = m

                cls_idx = int(lb_b[best_g].item()) - 1  # 0-indexed class
                if cls_idx < 0 or cls_idx >= K:
                    continue

                pred_mask = mask_logits[flat_idx, cls_idx]  # (mH, mW)

                if gt_masks_tgt is not None:
                    # Crop GT mask to proposal ROI and resize to (mH, mW)
                    x1 = max(0, int(float(props[n, 0].item())))
                    y1 = max(0, int(float(props[n, 1].item())))
                    x2 = max(x1 + 1, int(float(props[n, 2].item())))
                    y2 = max(y1 + 1, int(float(props[n, 3].item())))

                    gt_mask_full: Tensor = gt_masks_tgt[best_g]  # (H, W)
                    gt_crop = (
                        gt_mask_full[y1:y2, x1:x2].unsqueeze(0).unsqueeze(0).float()
                    )
                    gt_resized: Tensor = (
                        F.interpolate(
                            gt_crop, size=(mH, mW), mode="bilinear", align_corners=False
                        )
                        .squeeze(0)
                        .squeeze(0)
                    )
                    gt_mask = (gt_resized > 0.5).float()
                else:
                    gt_mask = lucid.zeros((mH, mW))

                all_losses.append(
                    F.binary_cross_entropy_with_logits(
                        pred_mask.reshape(-1), gt_mask.reshape(-1)
                    )
                )

            proposal_idx += N_i

        if not all_losses:
            return lucid.zeros((1,))
        return lucid.cat([l.reshape(1) for l in all_losses]).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> InstanceSegmentationOutput:
        """Run Mask R-CNN.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional training targets (list of dicts with
                     "boxes", "labels", and optionally "masks").

        Returns:
            ``InstanceSegmentationOutput``:
              ``logits``     : (Σ proposals, K+1) class logits.
              ``pred_boxes`` : (Σ proposals, 4) decoded xyxy boxes.
              ``pred_masks`` : (Σ proposals, K, 28, 28) mask logits.
              ``loss``       : total loss when targets provided.
        """
        B = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # 1. Backbone → [C2, C3, C4, C5]
        c_maps: list[Tensor] = cast(list[Tensor], self.backbone(x))

        # 2. FPN → [P2, P3, P4, P5, P6]
        fpn_feats: list[Tensor] = self.fpn.forward(c_maps)

        # Split: detection uses P2–P5, RPN uses all 5
        det_feats = fpn_feats[:4]  # [P2, P3, P4, P5]
        rpn_feats = fpn_feats  # [P2, P3, P4, P5, P6]

        # Strides for each RPN level
        rpn_strides: list[tuple[int, int]] = [
            (4, 4),
            (8, 8),
            (16, 16),
            (32, 32),
            (64, 64),
        ]

        # 3. Generate anchors
        anchors = self._anchor_gen.forward(rpn_feats, (iH, iW), rpn_strides)

        # 4. RPN → proposals
        proposals, _ = self._rpn.forward(rpn_feats, anchors, (iH, iW))

        # 5. FPN-level assignment for detection RoI Align
        level_assignments = _assign_fpn_levels(proposals, k0=4, num_levels=4)

        # 6. Detection RoI Align (7×7)
        det_crops = _fpn_roi_align(
            det_feats,
            proposals,
            level_assignments,
            output_size=self._cfg.roi_det_size,
            spatial_scales=self._det_spatial_scales,
        )

        # 7. Detection head
        K = self._cfg.num_classes
        if int(det_crops.shape[0]) > 0:
            all_logits, all_deltas = cast(
                tuple[Tensor, Tensor], self.det_head(det_crops)
            )
        else:
            all_logits = lucid.zeros((0, K + 1))
            all_deltas = lucid.zeros((0, K * 4))

        # 8. Mask RoI Align (14×14)
        mask_crops = _fpn_roi_align(
            det_feats,
            proposals,
            level_assignments,
            output_size=self._cfg.roi_mask_size,
            spatial_scales=self._det_spatial_scales,
        )

        # 9. Mask head
        if int(mask_crops.shape[0]) > 0:
            mask_logits: Tensor = cast(Tensor, self.mask_head(mask_crops))
        else:
            mask_logits = lucid.zeros((0, K, 28, 28))

        # 10. Decode detection boxes
        pred_boxes = self._decode_boxes(proposals, all_deltas, (iH, iW))

        # 11. Losses
        loss: Tensor | None = None
        if targets is not None:
            loss_rpn = self._rpn_loss(
                fpn_feats, (iH, iW), targets, rpn_strides, anchors
            )
            loss_det, fg_flags = self._det_loss(
                proposals, all_logits, all_deltas, targets
            )
            loss_mask = self._mask_loss(
                mask_crops, proposals, targets, fg_flags, level_assignments
            )
            loss = loss_rpn + loss_det + loss_mask

        return InstanceSegmentationOutput(
            logits=all_logits,
            pred_boxes=pred_boxes,
            pred_masks=mask_logits,
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
        flat_props = lucid.cat([p for p in proposals if int(p.shape[0]) > 0], dim=0)
        N = int(all_deltas.shape[0])
        if N == 0:
            return lucid.zeros((0, 4))
        K = self._cfg.num_classes
        top_deltas = all_deltas.reshape(N, K, 4)[:, 0, :]
        boxes = decode_boxes(top_deltas, flat_props, self._cfg.bbox_reg_weights)
        return clip_boxes_to_image(boxes, image_size)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: InstanceSegmentationOutput,
        proposals: list[Tensor],
    ) -> list[dict[str, Tensor]]:
        """Per-class NMS + mask binarisation.

        Returns list of per-image result dicts with
        "boxes", "scores", "labels", and "masks".
        """
        logits = output.logits
        pred_boxes = output.pred_boxes
        pred_masks = output.pred_masks
        K = self._cfg.num_classes
        results: list[dict[str, Tensor]] = []
        offset = 0

        for props in proposals:
            N_i = int(props.shape[0])
            lg_i = logits[offset : offset + N_i]
            bx_i = pred_boxes[offset : offset + N_i]
            mk_i = pred_masks[offset : offset + N_i]  # (N_i, K, 28, 28)
            offset += N_i

            scores_i = F.softmax(lg_i, dim=-1)
            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []
            keep_masks: list[Tensor] = []

            for c in range(1, K + 1):
                sc_c_all = scores_i[:, c]
                mask: list[int] = [
                    i
                    for i in range(N_i)
                    if float(sc_c_all[i].item()) >= self._cfg.score_thresh
                ]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask).long()
                sc_c = sc_c_all[mask_t]
                bx_c = bx_i[mask_t]
                mk_c = F.sigmoid(mk_i[mask_t, c - 1])  # (k, 28, 28)

                keep = batched_nms(
                    bx_c,
                    sc_c,
                    lucid.zeros(int(sc_c.shape[0])),
                    self._cfg.nms_thresh,
                )
                keep = keep[: self._cfg.max_detections]
                keep_boxes.append(bx_c[keep])
                keep_scores.append(sc_c[keep])
                keep_labels.append(lucid.full((int(keep.shape[0]),), float(c)))
                keep_masks.append((mk_c[keep] > self._cfg.mask_thresh).float())

            if keep_boxes:
                results.append(
                    {
                        "boxes": lucid.cat(keep_boxes, dim=0),
                        "scores": lucid.cat(keep_scores, dim=0),
                        "labels": lucid.cat(keep_labels, dim=0),
                        "masks": lucid.cat(keep_masks, dim=0),
                    }
                )
            else:
                results.append(
                    {
                        "boxes": lucid.zeros((0, 4)),
                        "scores": lucid.zeros((0,)),
                        "labels": lucid.zeros((0,)),
                        "masks": lucid.zeros((0, 28, 28)),
                    }
                )
        return results
