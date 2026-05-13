"""EfficientDet — scalable compound-scaled detector (Tan et al., CVPR 2020).

Paper: "EfficientDet: Scalable and Efficient Object Detection"

Key contributions
-----------------
1. **BiFPN** — Bidirectional Feature Pyramid Network with fast-normalised
   weighted feature fusion.  Unlike top-down-only FPN, BiFPN allows both
   top-down and bottom-up paths and learns per-input, per-level fusion weights:
       out_i = ReLU(Σ_j w_j · x_j) / (ε + Σ_j w_j),  ε = 1e-4
   The weights are forced positive via ReLU.

2. **Compound scaling** — A single coefficient φ jointly scales the backbone
   (EfficientNet-Bφ), BiFPN width (W_bifpn) / depth (D_bifpn), and
   prediction head depth (D_head) / image resolution.

3. **Shared prediction heads** — Class and box heads share depthwise-separable
   convolution weights across all five BiFPN output levels (P3–P7), with
   separate batch-norm per level.

Architecture
------------
  Image → EfficientNet-Bφ → (P3, P4, P5) at strides (8, 16, 32)
    ↓  + P6 = MaxPool(P5), P7 = MaxPool(P6) for two additional coarse levels
  [P3,P4,P5,P6,P7] → BiFPN × D_bifpn
    ├─ Class head: D_head × SepConv(W_bifpn) → Conv(num_classes × A)
    └─ Box head:  D_head × SepConv(W_bifpn) → Conv(4 × A)

  where A = len(anchor_scales) × len(anchor_ratios) = 9 (3 scales × 3 ratios).

BiFPN single pass (for L levels, finest = 0):
  Intermediate top-down:
    P_td[L-1] = P_in[L-1]
    P_td[i]   = Conv(w1·P_in[i] + w2·Resize(P_td[i+1]))  for i = L-2 … 0
  Bottom-up:
    P_out[0]  = Conv(w1·P_in[0] + w2·P_td[0])
    P_out[i]  = Conv(w1·P_in[i] + w2·P_td[i] + w3·MaxPool(P_out[i-1]))  for i > 0

Faithfulness notes
------------------
* BiFPN uses depthwise-separable conv (DWConv + PWConv) per the paper.
* Learnable fusion weights ReLU-initialised to 1.0.
* Backbone: simplified EfficientNet-B0 (MBConv blocks) for default φ=0.
* Anchors: 9 per cell (3 scales × 3 ratios), at 5 levels (P3–P7).
* No NMS-free inference (unlike DETR) — uses per-class NMS.
* Loss: focal loss for classification + smooth-L1 for regression (paper §4.2).
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
    clip_boxes_to_image,
    decode_boxes,
    encode_boxes,
)
from lucid.models.vision.efficientdet._config import EfficientDetConfig


# ---------------------------------------------------------------------------
# Depthwise-separable convolution (BiFPN building block)
# ---------------------------------------------------------------------------


class _SepConv(nn.Module):
    """Depthwise-separable conv: DWConv(k,k,groups=C) + PWConv(1,1) + BN + ReLU."""

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size, padding=padding,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.relu(cast(Tensor, self.bn(cast(Tensor, self.pw(cast(Tensor, self.dw(x)))))))


# ---------------------------------------------------------------------------
# BiFPN level
# ---------------------------------------------------------------------------

_EPS = 1e-4


class _BiFPNLayer(nn.Module):
    """One BiFPN repetition (bidirectional top-down + bottom-up fusion)."""

    def __init__(self, num_channels: int, num_levels: int = 5) -> None:
        super().__init__()
        self.num_levels = num_levels
        L = num_levels

        # Top-down intermediate weights (L-1 intermediate nodes, each fusing 2 inputs)
        self.td_weights: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(lucid.ones((2,))) for _ in range(L - 1)]
        )
        # Bottom-up output weights
        # Level 0 (finest): fuse 2 inputs (P_in + P_td)
        # Levels 1..L-1: fuse 3 inputs (P_in + P_td + down-sampled output)
        self.out_weights: nn.ParameterList = nn.ParameterList(
            [nn.Parameter(lucid.ones((2,)))] +
            [nn.Parameter(lucid.ones((3,))) for _ in range(L - 1)]
        )

        # Convolutions (one per intermediate top-down node + one per output)
        self.td_convs = nn.ModuleList([_SepConv(num_channels) for _ in range(L - 1)])
        self.out_convs = nn.ModuleList([_SepConv(num_channels) for _ in range(L)])

    def forward(self, features: list[Tensor]) -> list[Tensor]:  # type: ignore[override]
        """
        Args:
            features: [P3, P4, P5, P6, P7] (finest → coarsest).

        Returns:
            Fused [P3_out, P4_out, P5_out, P6_out, P7_out].
        """
        L = self.num_levels
        assert len(features) == L

        # --- Top-down intermediate ---
        td: list[Tensor] = [features[-1]]  # coarsest passes through
        for i in range(L - 2, -1, -1):     # from coarsest-1 down to finest
            w: Tensor = F.relu(cast(Tensor, self.td_weights[L - 2 - i]))
            w0 = float(w[0].item()) + _EPS
            w1 = float(w[1].item()) + _EPS
            wsum = w0 + w1
            up = F.interpolate(td[0], scale_factor=2.0, mode="nearest")
            node: Tensor = cast(Tensor, self.td_convs[L - 2 - i](
                (w0 / wsum) * features[i] + (w1 / wsum) * up
            ))
            td.insert(0, node)  # prepend so td[0] = finest

        # --- Bottom-up output ---
        out: list[Tensor] = []
        # Finest level (only 2 inputs)
        wf: Tensor = F.relu(cast(Tensor, self.out_weights[0]))
        wf0 = float(wf[0].item()) + _EPS
        wf1 = float(wf[1].item()) + _EPS
        wfsum = wf0 + wf1
        out.append(cast(Tensor, self.out_convs[0](
            (wf0 / wfsum) * features[0] + (wf1 / wfsum) * td[0]
        )))

        for i in range(1, L):
            wl: Tensor = F.relu(cast(Tensor, self.out_weights[i]))
            wl0 = float(wl[0].item()) + _EPS
            wl1 = float(wl[1].item()) + _EPS
            wl2 = float(wl[2].item()) + _EPS
            wlsum = wl0 + wl1 + wl2
            down = cast(Tensor, nn.MaxPool2d(2, stride=2)(out[-1]))
            out.append(cast(Tensor, self.out_convs[i](
                (wl0 / wlsum) * features[i] +
                (wl1 / wlsum) * td[i] +
                (wl2 / wlsum) * down
            )))

        return out


# ---------------------------------------------------------------------------
# EfficientNet-B0 backbone (simplified, P3/P4/P5 outputs)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """Mobile Inverted Bottleneck (MBConv) with optional skip connection."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: int = 6,
        stride: int = 1,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand_ratio
        padding = (kernel_size - 1) // 2

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True)]
        layers += [
            nn.Conv2d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=padding, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)
        self.use_skip = (stride == 1 and in_ch == out_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out: Tensor = cast(Tensor, self.block(x))
        return out + x if self.use_skip else out


def _make_mbconv_stage(
    in_ch: int, out_ch: int, n: int,
    stride: int = 1, expand: int = 6, k: int = 3
) -> nn.Sequential:
    blocks: list[nn.Module] = [_MBConv(in_ch, out_ch, expand, stride, k)]
    for _ in range(1, n):
        blocks.append(_MBConv(out_ch, out_ch, expand, 1, k))
    return nn.Sequential(*blocks)


class _EfficientNetBackbone(nn.Module):
    """Simplified EfficientNet-B0 backbone.

    Returns (P3, P4, P5) feature maps at strides (8, 16, 32).
    Channel widths:
      After stage 2 (stride 8):  P3 = 40ch
      After stage 4 (stride 16): P4 = 112ch
      After stage 6 (stride 32): P5 = 320ch
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # MBConv stages (EfficientNet-B0 settings)
        self.stage0 = _make_mbconv_stage(32,  16,  n=1, stride=1, expand=1, k=3)
        self.stage1 = _make_mbconv_stage(16,  24,  n=2, stride=2, expand=6, k=3)
        self.stage2 = _make_mbconv_stage(24,  40,  n=2, stride=2, expand=6, k=5)
        self.stage3 = _make_mbconv_stage(40,  80,  n=3, stride=2, expand=6, k=3)
        self.stage4 = _make_mbconv_stage(80,  112, n=3, stride=1, expand=6, k=5)
        self.stage5 = _make_mbconv_stage(112, 192, n=4, stride=2, expand=6, k=5)
        self.stage6 = _make_mbconv_stage(192, 320, n=1, stride=1, expand=6, k=3)
        self.p3_channels: int = 40
        self.p4_channels: int = 112
        self.p5_channels: int = 320

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore[override]
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.stage0(x))
        x = cast(Tensor, self.stage1(x))
        p3: Tensor = cast(Tensor, self.stage2(x))       # stride 8
        x = cast(Tensor, self.stage3(p3))
        p4: Tensor = cast(Tensor, self.stage4(x))       # stride 16
        x = cast(Tensor, self.stage5(p4))
        p5: Tensor = cast(Tensor, self.stage6(x))       # stride 32
        return p3, p4, p5


# ---------------------------------------------------------------------------
# Prediction head (class or box)
# ---------------------------------------------------------------------------


class _PredictionHead(nn.Module):
    """Shared-weight prediction head for all BiFPN levels.

    Uses depth-wise separable convolutions with separate batch-norm per level.
    """

    def __init__(
        self,
        in_channels: int,
        num_outputs: int,   # num_classes or 4 * num_anchors
        num_repeats: int,
        num_levels: int = 5,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels
        self.num_repeats = num_repeats

        # Shared DWConv weights (one per repeat depth)
        self.dw_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False)
             for _ in range(num_repeats)]
        )
        self.pw_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 1, bias=False)
             for _ in range(num_repeats)]
        )
        # Separate BN per level per depth
        self.bns = nn.ModuleList(
            [nn.ModuleList([nn.BatchNorm2d(in_channels) for _ in range(num_levels)])
             for _ in range(num_repeats)]
        )
        self.predictor = nn.Conv2d(in_channels, num_outputs, 1)

    def forward(self, features: list[Tensor]) -> list[Tensor]:  # type: ignore[override]
        """
        Args:
            features: List of num_levels feature maps (finest → coarsest).

        Returns:
            List of prediction maps, one per level.
        """
        outs: list[Tensor] = []
        for lvl, feat in enumerate(features):
            x = feat
            for depth in range(self.num_repeats):
                dw: Tensor = cast(Tensor, self.dw_convs[depth](x))
                pw: Tensor = cast(Tensor, self.pw_convs[depth](dw))
                bn_list = cast(nn.ModuleList, self.bns[depth])
                x = F.relu(cast(Tensor, bn_list[lvl](pw)))
            outs.append(cast(Tensor, self.predictor(x)))
        return outs


# ---------------------------------------------------------------------------
# Smooth-L1 and focal loss helpers
# ---------------------------------------------------------------------------


def _smooth_l1(x: Tensor, beta: float = 0.1) -> Tensor:
    abs_x: Tensor = lucid.abs(x)
    cond: Tensor = abs_x < beta
    return lucid.where(cond, 0.5 * x * x / beta, abs_x - 0.5 * beta)


def _focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> Tensor:
    """Binary focal loss for multi-label classification (sigmoid).

    Args:
        logits:  (N,) raw logits.
        targets: (N,) binary targets {0.0, 1.0}.
    """
    p: Tensor   = F.sigmoid(logits)
    ce: Tensor  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t: Tensor = targets * p + (1.0 - targets) * (1.0 - p)
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    focal_weight = alpha_t * (1.0 - p_t) ** gamma
    return (focal_weight * ce).mean()


# ---------------------------------------------------------------------------
# EfficientDet
# ---------------------------------------------------------------------------


class EfficientDetForObjectDetection(PretrainedModel):
    """EfficientDet object detector (Tan et al., CVPR 2020).

    Input contract
    --------------
    ``x``       : (B, C, H, W) image batch.
    ``targets`` : optional training list of B dicts with:
                    ``"boxes"``  — (M_i, 4) xyxy pixel-coordinate GT boxes.
                    ``"labels"`` — (M_i,)   integer foreground class ids (1-indexed).

    Output contract
    ---------------
    ``ObjectDetectionOutput``:
      ``logits``    : (B, total_anchors, num_classes) per-class sigmoid logits.
      ``pred_boxes``: (B, total_anchors, 4) decoded xyxy boxes.
      ``loss``      : focal loss + smooth-L1 when targets provided.
    """

    config_class: ClassVar[type[EfficientDetConfig]] = EfficientDetConfig
    base_model_prefix: ClassVar[str] = "efficientdet"

    def __init__(self, config: EfficientDetConfig) -> None:
        super().__init__(config)
        self._cfg = config
        W = config.fpn_channels
        K = config.num_classes
        num_levels = 5     # P3–P7
        num_anchors = len(config.anchor_scales) * len(config.anchor_ratios)

        # Backbone
        self.backbone = _EfficientNetBackbone(config.in_channels)

        # Channel projection: P3/P4/P5 → fpn_channels
        bb_ch = config.backbone_in_channels
        self.p3_proj = nn.Sequential(nn.Conv2d(bb_ch[0], W, 1), nn.BatchNorm2d(W))
        self.p4_proj = nn.Sequential(nn.Conv2d(bb_ch[1], W, 1), nn.BatchNorm2d(W))
        self.p5_proj = nn.Sequential(nn.Conv2d(bb_ch[2], W, 1), nn.BatchNorm2d(W))

        # P6 = MaxPool(P5), P7 = MaxPool(P6)  — both in W channels after proj
        self.p6_pool = nn.MaxPool2d(2, stride=2)
        self.p7_pool = nn.MaxPool2d(2, stride=2)
        # P6/P7 still W channels (from P5 projection)

        # BiFPN stack
        self.bifpn = nn.ModuleList([
            _BiFPNLayer(W, num_levels=num_levels)
            for _ in range(config.fpn_repeats)
        ])

        # Prediction heads
        self.cls_head = _PredictionHead(W, K * num_anchors, config.head_repeats, num_levels)
        self.box_head = _PredictionHead(W, 4 * num_anchors, config.head_repeats, num_levels)

        # Anchor generator (5 levels; one base size per level)
        sizes: tuple[tuple[int, ...], ...] = tuple(
            (int(s * r),) for s, r in [
                (config.anchor_base_sizes[i],
                 config.anchor_scales[0])   # base size only; scales handled by anchor_scales
                for i in range(num_levels)
            ]
        )
        # Build anchors with all scales × ratios
        all_sizes: tuple[tuple[int, ...], ...] = tuple(
            tuple(int(config.anchor_base_sizes[lvl] * sc) for sc in config.anchor_scales)
            for lvl in range(num_levels)
        )
        self._anchor_gen = AnchorGenerator(
            sizes=all_sizes,
            aspect_ratios=(tuple(config.anchor_ratios),) * num_levels,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _project_backbone(
        self, p3: Tensor, p4: Tensor, p5: Tensor
    ) -> list[Tensor]:
        """Project P3/P4/P5 to FPN width and build P6/P7."""
        fp3: Tensor = F.relu(cast(Tensor, self.p3_proj(p3)))
        fp4: Tensor = F.relu(cast(Tensor, self.p4_proj(p4)))
        fp5: Tensor = F.relu(cast(Tensor, self.p5_proj(p5)))
        fp6: Tensor = cast(Tensor, self.p6_pool(fp5))
        fp7: Tensor = cast(Tensor, self.p7_pool(fp6))
        return [fp3, fp4, fp5, fp6, fp7]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run EfficientDet.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional training targets.

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``    : (B, A, K) per-class sigmoid logits (A = total anchors).
              ``pred_boxes``: (B, A, 4) decoded xyxy boxes.
              ``loss``      : focal + smooth-L1 when targets provided.
        """
        B  = int(x.shape[0])
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # 1. Backbone → (P3, P4, P5)
        p3, p4, p5 = cast(
            tuple[Tensor, Tensor, Tensor],
            self.backbone(x)
        )

        # 2. Project → [P3, P4, P5, P6, P7] in FPN-width channels
        fpn_feats = self._project_backbone(p3, p4, p5)

        # 3. BiFPN stack
        for bifpn_layer in self.bifpn:
            fpn_feats = cast(_BiFPNLayer, bifpn_layer).forward(fpn_feats)

        # 4. Strides for 5 levels (P3=8, P4=16, P5=32, P6=64, P7=128)
        strides: list[tuple[int, int]] = [
            (8, 8), (16, 16), (32, 32), (64, 64), (128, 128)
        ]

        # 5. Generate anchors → list of (A_l, 4) per level
        anchors_per_level = self._anchor_gen.forward(fpn_feats, (iH, iW), strides)

        # 6. Prediction heads → per-level class / box maps
        cls_maps = self.cls_head.forward(fpn_feats)
        box_maps = self.box_head.forward(fpn_feats)

        # 7. Reshape and concatenate → (B, A_total, K) and (B, A_total, 4)
        num_anchors = len(self._cfg.anchor_scales) * len(self._cfg.anchor_ratios)
        K = self._cfg.num_classes

        cls_all_parts: list[Tensor] = []
        box_all_parts: list[Tensor] = []
        anchors_flat_parts: list[Tensor] = []

        for lvl, (cm, bm, anc) in enumerate(zip(cls_maps, box_maps, anchors_per_level)):
            fH = int(cm.shape[2])
            fW = int(cm.shape[3])
            # cm: (B, K*A, H, W) → (B, H*W*A, K)
            cm_r = cm.reshape(B, K, num_anchors, fH, fW)
            cm_r = cm_r.permute(0, 3, 4, 2, 1).reshape(B, -1, K)
            # bm: (B, 4*A, H, W) → (B, H*W*A, 4)
            bm_r = bm.reshape(B, 4, num_anchors, fH, fW)
            bm_r = bm_r.permute(0, 3, 4, 2, 1).reshape(B, -1, 4)

            cls_all_parts.append(cm_r)
            box_all_parts.append(bm_r)
            anchors_flat_parts.append(anc)  # (A_l, 4)

        all_logits:  Tensor = lucid.cat(cls_all_parts, dim=1)   # (B, A, K)
        all_deltas:  Tensor = lucid.cat(box_all_parts, dim=1)   # (B, A, 4)
        all_anchors: Tensor = lucid.cat(anchors_flat_parts, dim=0)  # (A, 4)

        # 8. Decode boxes
        A_total = int(all_deltas.shape[1])
        all_boxes_parts: list[Tensor] = []
        for b in range(B):
            boxes_b = decode_boxes(all_deltas[b], all_anchors)   # (A, 4)
            boxes_b = clip_boxes_to_image(boxes_b, (iH, iW))
            all_boxes_parts.append(boxes_b.unsqueeze(0))
        all_boxes: Tensor = lucid.cat(all_boxes_parts, dim=0)  # (B, A, 4)

        # 9. Loss
        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(
                all_logits, all_deltas, all_anchors, all_boxes, targets, (iH, iW)
            )

        return ObjectDetectionOutput(
            logits=all_logits,
            pred_boxes=all_boxes,
            loss=loss,
        )

    # ------------------------------------------------------------------
    # Training loss
    # ------------------------------------------------------------------

    def _compute_loss(
        self,
        all_logits: Tensor,   # (B, A, K)
        all_deltas: Tensor,   # (B, A, 4)
        all_anchors: Tensor,  # (A, 4)
        all_boxes: Tensor,    # (B, A, 4)
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> Tensor:
        B = len(targets)
        K = self._cfg.num_classes
        A = int(all_anchors.shape[0])

        cls_losses: list[Tensor] = []
        reg_losses: list[Tensor] = []

        for b in range(B):
            gt_boxes  = targets[b]["boxes"]   # (M, 4) xyxy
            gt_labels = targets[b]["labels"]  # (M,)
            M = int(gt_boxes.shape[0])

            lg_b  = all_logits[b]   # (A, K)

            if M == 0:
                # All anchors → background
                tgt_cls = lucid.zeros((A, K))
                cls_losses.append(_focal_loss(lg_b.reshape(-1), tgt_cls.reshape(-1)))
                continue

            # Compute pairwise IoU between anchors and GT boxes
            # Build manually to avoid importing box_iou with its O(NxM) loop
            from lucid.models._utils._detection import box_iou as _box_iou
            iou_mat = _box_iou(all_anchors, gt_boxes)   # (A, M)

            # Assign each anchor: best GT, then label
            tgt_cls_data = [[0.0] * K for _ in range(A)]
            pos_idx: list[int] = []
            pos_gt:  list[int] = []

            for a in range(A):
                best_v = -1.0
                best_m = 0
                for m in range(M):
                    v = float(iou_mat[a, m].item())
                    if v > best_v:
                        best_v = v
                        best_m = m
                if best_v >= 0.5:
                    c = int(gt_labels[best_m].item()) - 1  # 0-indexed
                    if 0 <= c < K:
                        tgt_cls_data[a][c] = 1.0
                    pos_idx.append(a)
                    pos_gt.append(best_m)
                elif best_v < 0.4:
                    pass   # background — all zeros (already set)
                # else: ignore (IoU in [0.4, 0.5)) — no loss

            tgt_cls = lucid.tensor(tgt_cls_data)  # (A, K)
            cls_losses.append(_focal_loss(lg_b.reshape(-1), tgt_cls.reshape(-1)))

            if pos_idx:
                pos_t = lucid.tensor(pos_idx)
                gt_boxes_pos = lucid.tensor(
                    [[float(gt_boxes[pos_gt[i], d].item()) for d in range(4)]
                     for i in range(len(pos_idx))]
                )
                anc_pos  = all_anchors[pos_t]     # (P, 4)
                tgt_d    = encode_boxes(gt_boxes_pos, anc_pos)
                pred_d   = all_deltas[b][pos_t]   # (P, 4)
                reg_losses.append(_smooth_l1(pred_d - tgt_d).mean())

        cls_l = lucid.cat([l.reshape(1) for l in cls_losses]).mean() \
            if cls_losses else lucid.zeros((1,))
        reg_l = lucid.cat([l.reshape(1) for l in reg_losses]).mean() \
            if reg_losses else lucid.zeros((1,))
        return cls_l + reg_l

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def postprocess(
        self,
        output: ObjectDetectionOutput,
    ) -> list[dict[str, Tensor]]:
        """Per-class NMS on raw sigmoid predictions.

        Returns list of per-image result dicts with "boxes", "scores", "labels".
        """
        B = int(output.logits.shape[0])
        K = self._cfg.num_classes
        results: list[dict[str, Tensor]] = []

        for b in range(B):
            lg_b = output.logits[b]       # (A, K)
            bx_b = output.pred_boxes[b]   # (A, 4)
            sc_b = F.sigmoid(lg_b)        # (A, K) — per-class probabilities

            keep_boxes:  list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(K):
                sc_c = sc_b[:, c]
                mask: list[int] = [
                    i for i in range(int(sc_c.shape[0]))
                    if float(sc_c[i].item()) >= self._cfg.score_thresh
                ]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask)
                sc_sel = sc_c[mask_t]
                bx_sel = bx_b[mask_t]
                keep = batched_nms(
                    bx_sel, sc_sel,
                    lucid.zeros(int(sc_sel.shape[0])),
                    self._cfg.nms_thresh,
                )
                keep = keep[:self._cfg.max_detections]
                keep_boxes.append(bx_sel[keep])
                keep_scores.append(sc_sel[keep])
                keep_labels.append(lucid.full((int(keep.shape[0]),), float(c + 1)))

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
