"""YOLOv4 — You Only Look Once v4 (Bochkovskiy et al., arXiv 2020).

Paper: "YOLOv4: Optimal Speed and Accuracy of Object Detection"
https://arxiv.org/abs/2004.10934

Key improvements over YOLOv3
-----------------------------
1. **CSPDarknet-53 backbone** — replaces standard Darknet-53 residual blocks
   with Cross Stage Partial (CSP) blocks that split the feature map, apply
   residuals to one branch, and concatenate, reducing gradient duplication and
   improving learning capacity.
2. **SPP module** (Spatial Pyramid Pooling) at P5: MaxPool(k=5), MaxPool(k=9),
   MaxPool(k=13) + original feature → concat, quadrupling the receptive field
   at essentially no extra inference cost.
3. **PANet neck** (Path Aggregation Network): augments the FPN-style top-down
   pathway with a bottom-up pathway, so each detection scale receives both
   fine-grained detail (from bottom-up) and semantic context (from top-down).
4. **CIoU loss** for bounding-box regression — Complete IoU accounts for
   overlap area, centre distance, and aspect-ratio consistency simultaneously,
   giving faster convergence and better localisation than MSE.

Architecture overview
---------------------
  Image (B, 3, H, W)
    ↓  CSPDarknet-53 backbone → P3 (256ch), P4 (512ch), P5 (1024ch)
    ↓  SPP at P5: MaxPool(5,9,13) concat → 2048ch → compress → 512ch
    ↓  PANet neck
       Top-down:
         P5(512) → upsample → concat P4(512) → CSP compress → P4'(256)
         P4'(256) → upsample → concat P3(256) → CSP compress → P3'(128)
       Bottom-up:
         P3'(128) → stride-2 conv → concat P4'(256) → CSP compress → P4''(256)
         P4''(256) → stride-2 conv → concat P5(512) → CSP compress → P5''(512)
    ↓  Detection heads at P3'', P4'', P5''
       Each: Conv → nA*(5+C) predictions
    ↓  Decode + CIoU loss / NMS

Loss (training)
---------------
  Requires ``targets`` — list of B dicts:
    "boxes"  : (M, 4) xyxy pixel coordinates
    "labels" : (M,)   integer class ids (0-indexed)

  Anchor assignment: same as YOLOv3 (best-IoU per GT at matching grid cell).
  L = L_ciou(box)   [positive anchors only]
    + L_bce(obj)    [positive=1, negative=lambda_noobj scaled]
    + L_bce(cls)    [positive anchors only]
"""

import math
from dataclasses import dataclass
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import ModelConfig, PretrainedModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._registry import register_model
from lucid.models._utils._detection import (
    batched_nms,
    box_iou,
    clip_boxes_to_image,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class YOLOV4Config(ModelConfig):
    """Configuration for YOLOv4.

    YOLOv4 (Bochkovskiy et al., 2020) uses CSPDarknet-53 as backbone,
    SPP + PANet as neck, and three detection scales at strides 8/16/32.

    Args:
        num_classes:  Number of foreground classes (COCO default = 80).
        in_channels:  Input image channels.
        anchors:      9 (width, height) anchor pairs in pixels, 3 per scale.
                      Order: P3 (small), P4 (medium), P5 (large).
        strides:      Feature-map strides for P3, P4, P5.
        score_thresh: Minimum class score to keep a detection.
        nms_thresh:   IoU threshold for per-class NMS.
        lambda_noobj: Objectness loss weight for negative anchors.
    """

    model_type: ClassVar[str] = "yolo_v4"

    num_classes: int = 80
    in_channels: int = 3

    anchors: tuple[tuple[float, float], ...] = (
        # P3 (stride 8) — small objects
        (10.0, 13.0),
        (16.0, 30.0),
        (33.0, 23.0),
        # P4 (stride 16) — medium objects
        (30.0, 61.0),
        (62.0, 45.0),
        (59.0, 119.0),
        # P5 (stride 32) — large objects
        (116.0, 90.0),
        (156.0, 198.0),
        (373.0, 326.0),
    )

    strides: tuple[int, int, int] = (8, 16, 32)
    score_thresh: float = 0.5
    nms_thresh: float = 0.5
    lambda_noobj: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(tuple(a) for a in self.anchors))
        object.__setattr__(self, "strides", tuple(self.strides))


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _ConvBnLeaky(nn.Module):
    """Conv2d(bias=False) → BatchNorm2d → LeakyReLU(0.1).

    Used by the YOLOv4 neck (SPP, PANet) and the prediction heads.  The
    backbone (CSPDarknet-53) uses ``_ConvBnMish`` instead — paper §3.4.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = -1,
    ) -> None:
        super().__init__()
        pad = padding if padding >= 0 else (kernel - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.leaky_relu(
            cast(Tensor, self.bn(cast(Tensor, self.conv(x)))), negative_slope=0.1
        )


class _ConvBnMish(nn.Module):
    """Conv2d(bias=False) → BatchNorm2d → Mish.

    YOLOv4 backbone activation per paper §3.4: ``Mish(x) = x · tanh(softplus(x))``.
    Empirically smoother gradient flow than LeakyReLU in deep CSP residual stacks.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = -1,
    ) -> None:
        super().__init__()
        pad = padding if padding >= 0 else (kernel - 1) // 2
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=pad, bias=False
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.Mish()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(
            Tensor, self.act(cast(Tensor, self.bn(cast(Tensor, self.conv(x)))))
        )


# ---------------------------------------------------------------------------
# CSP Block — Cross Stage Partial
# ---------------------------------------------------------------------------


class _CSPBottleneck(nn.Module):
    """One CSP bottleneck unit (1×1 → 3×3) used inside _CSPBlock.

    Lives inside the CSPDarknet-53 backbone → uses Mish activation per paper §3.4.
    """

    def __init__(self, ch: int) -> None:
        super().__init__()
        mid = ch // 2
        self.conv1 = _ConvBnMish(ch, mid, 1)
        self.conv2 = _ConvBnMish(mid, ch, 3)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x + cast(Tensor, self.conv2(cast(Tensor, self.conv1(x))))


class _CSPBlock(nn.Module):
    """CSP block: split input into two routes, apply residuals to one, concat.

    Route 1 (skip): Conv(in_ch, in_ch//2, 1) — direct pass-through.
    Route 2 (main): Conv(in_ch, in_ch//2, 1) → n_repeats × _CSPBottleneck.
    Merge: concat(route1, route2) → Conv(in_ch, in_ch, 1).

    All convs inside the backbone use Mish; the same primitive is reused by
    the neck (PANet, SPP) where it's instantiated with ``act="leaky"``.

    Args:
        in_ch:      Number of input channels.
        n_repeats:  Number of residual bottleneck repeats in route 2.
        act:        ``"mish"`` for the backbone, ``"leaky"`` for the neck.
    """

    def __init__(self, in_ch: int, n_repeats: int, act: str = "mish") -> None:
        super().__init__()
        half = in_ch // 2
        Conv = _ConvBnMish if act == "mish" else _ConvBnLeaky
        self.route1 = Conv(in_ch, half, 1)  # skip branch
        self.route2 = Conv(in_ch, half, 1)  # main branch
        # Bottlenecks always use the same activation as the surrounding block.
        self.bottlenecks = nn.Sequential(
            *[_CSPBottleneck(half) for _ in range(n_repeats)]
        )
        self.merge = Conv(in_ch, in_ch, 1)  # after concat

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        r1 = cast(Tensor, self.route1(x))
        r2 = cast(Tensor, self.bottlenecks(cast(Tensor, self.route2(x))))
        merged = lucid.cat([r1, r2], dim=1)  # (B, in_ch, H, W)
        return cast(Tensor, self.merge(merged))


# ---------------------------------------------------------------------------
# CSPDarknet-53 backbone
# ---------------------------------------------------------------------------


class _CSPDarknet53(nn.Module):
    """CSPDarknet-53 backbone.

    Returns three feature maps:
      P3 : (B, 256,  H/8,  W/8)
      P4 : (B, 512,  H/16, W/16)
      P5 : (B, 1024, H/32, W/32)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stem + every stride-2 down conv + every CSP block use Mish (paper §3.4).
        self.stem = _ConvBnMish(in_channels, 32, 3)

        # Stage 1: stride-2 → 64ch, 1×CSP
        self.down1 = _ConvBnMish(32, 64, 3, stride=2)
        self.csp1 = _CSPBlock(64, 1, act="mish")

        # Stage 2: stride-2 → 128ch, 2×CSP
        self.down2 = _ConvBnMish(64, 128, 3, stride=2)
        self.csp2 = _CSPBlock(128, 2, act="mish")

        # Stage 3: stride-2 → 256ch, 8×CSP  [P3]
        self.down3 = _ConvBnMish(128, 256, 3, stride=2)
        self.csp3 = _CSPBlock(256, 8, act="mish")

        # Stage 4: stride-2 → 512ch, 8×CSP  [P4]
        self.down4 = _ConvBnMish(256, 512, 3, stride=2)
        self.csp4 = _CSPBlock(512, 8, act="mish")

        # Stage 5: stride-2 → 1024ch, 4×CSP  [P5]
        self.down5 = _ConvBnMish(512, 1024, 3, stride=2)
        self.csp5 = _CSPBlock(1024, 4, act="mish")

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore[override]
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.csp1(cast(Tensor, self.down1(x))))
        x = cast(Tensor, self.csp2(cast(Tensor, self.down2(x))))
        x = cast(Tensor, self.csp3(cast(Tensor, self.down3(x))))
        p3 = x  # (B, 256,  H/8,  W/8)
        x = cast(Tensor, self.csp4(cast(Tensor, self.down4(x))))
        p4 = x  # (B, 512,  H/16, W/16)
        x = cast(Tensor, self.csp5(cast(Tensor, self.down5(x))))
        p5 = x  # (B, 1024, H/32, W/32)
        return p3, p4, p5


# ---------------------------------------------------------------------------
# SPP module (Spatial Pyramid Pooling)
# ---------------------------------------------------------------------------


class _SPP(nn.Module):
    """Spatial Pyramid Pooling module used at P5.

    Pools input with MaxPool(k=5), MaxPool(k=9), MaxPool(k=13), then
    concatenates with the original feature map.  This 4× channel expansion
    is followed by a compress Conv to restore the original channel count.

    Args:
        in_ch:  Input channels (1024 for P5).
        out_ch: Output channels (512 — half of in_ch).
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        # Pre-SPP compress: in_ch → in_ch//2
        half = in_ch // 2
        self.pre = nn.Sequential(
            _ConvBnLeaky(in_ch, half, 1),
            _ConvBnLeaky(half, in_ch, 3),
            _ConvBnLeaky(in_ch, half, 1),
        )
        # MaxPool at 3 kernel sizes (same-padding to preserve spatial dims)
        self.pool5 = nn.MaxPool2d(5, stride=1, padding=2)
        self.pool9 = nn.MaxPool2d(9, stride=1, padding=4)
        self.pool13 = nn.MaxPool2d(13, stride=1, padding=6)
        # Post-SPP: 4×half → out_ch
        self.post = nn.Sequential(
            _ConvBnLeaky(half * 4, in_ch, 1),
            _ConvBnLeaky(in_ch, out_ch, 3),
            _ConvBnLeaky(out_ch, out_ch, 1),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.pre(x))  # (B, half, H, W)
        p5 = cast(Tensor, self.pool5(x))
        p9 = cast(Tensor, self.pool9(x))
        p13 = cast(Tensor, self.pool13(x))
        concat = lucid.cat([x, p5, p9, p13], dim=1)  # (B, 4*half, H, W)
        return cast(Tensor, self.post(concat))  # (B, out_ch, H, W)


# ---------------------------------------------------------------------------
# PANet neck
# ---------------------------------------------------------------------------


class _PANetNeck(nn.Module):
    """Path Aggregation Network neck.

    Connects the CSPDarknet-53 backbone outputs (P3, P4, P5) through:
      1. SPP at P5.
      2. Top-down pathway: P5→P4'→P3' (FPN-style, with CSP blocks).
      3. Bottom-up pathway: P3'→P4''→P5'' (PAN-style, with CSP blocks).

    Output channels:
      P3'' : 128ch
      P4'' : 256ch
      P5'' : 512ch
    """

    def __init__(self) -> None:
        super().__init__()
        # SPP at P5 (1024→512ch)
        self.spp = _SPP(1024, 512)

        # Top-down: P5(512) → upsample → concat P4(512) → P4'(512)
        # Lateral compress on P5 to 256ch before upsampling
        self.p5_lateral = _ConvBnLeaky(512, 256, 1)
        # Lateral compress on P4 from 512 to 256ch so concat = 512ch
        self.p4_lateral = _ConvBnLeaky(512, 256, 1)
        self.p4_csp_td = _CSPBlock(512, 2)  # 256+256 concat → 512ch, out=512ch
        self.p4_td_lat = _ConvBnLeaky(512, 128, 1)

        # Top-down: P4'(512→128) → upsample → concat P3(256→128) → P3'(256)
        self.p3_lateral = _ConvBnLeaky(256, 128, 1)  # compress P3 from 256 to 128
        self.p3_csp_td = _CSPBlock(256, 2)  # 128+128 concat → 256ch, out=256ch

        # Bottom-up: P3'(256) → stride-2 conv → concat P4'(128) → P4''(256)
        self.p3_down = _ConvBnLeaky(256, 128, 3, stride=2)
        self.p4_csp_bu = _CSPBlock(256, 2)  # 128+128 → 256ch

        # Bottom-up: P4''(256) → stride-2 conv → concat P5(256) → P5''(512)
        self.p4_down = _ConvBnLeaky(256, 256, 3, stride=2)
        self.p5_csp_bu = _CSPBlock(512, 2)  # 256+256 → 512ch

    def forward(  # type: ignore[override]
        self,
        p3: Tensor,
        p4: Tensor,
        p5: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run PANet.

        Args:
            p3: (B, 256,  H/8,  W/8)
            p4: (B, 512,  H/16, W/16)
            p5: (B, 1024, H/32, W/32)

        Returns:
            (p3_out, p4_out, p5_out) — feature maps for detection heads.
        """
        # SPP at P5
        p5_spp = cast(Tensor, self.spp(p5))  # (B, 512, H/32, W/32)

        # Top-down P5→P4'
        p5_lat = cast(Tensor, self.p5_lateral(p5_spp))  # (B, 256, H/32, W/32)
        fH4 = int(p4.shape[2])
        fW4 = int(p4.shape[3])
        p5_up = F.interpolate(
            p5_lat, size=(fH4, fW4), mode="nearest"
        )  # (B, 256, H/16, W/16)
        p4_comp = cast(Tensor, self.p4_lateral(p4))  # (B, 256, H/16, W/16)
        p4_cat = lucid.cat([p5_up, p4_comp], dim=1)  # (B, 512, H/16, W/16)
        p4_td = cast(Tensor, self.p4_csp_td(p4_cat))  # (B, 512, H/16, W/16)

        # Top-down P4'→P3'
        p4_lat = cast(Tensor, self.p4_td_lat(p4_td))  # (B, 128, H/16, W/16)
        fH3 = int(p3.shape[2])
        fW3 = int(p3.shape[3])
        p4_up = F.interpolate(
            p4_lat, size=(fH3, fW3), mode="nearest"
        )  # (B, 128, H/8, W/8)
        p3_lat = cast(Tensor, self.p3_lateral(p3))  # (B, 128, H/8, W/8)
        p3_cat = lucid.cat([p4_up, p3_lat], dim=1)  # (B, 256, H/8, W/8)
        p3_td = cast(Tensor, self.p3_csp_td(p3_cat))  # (B, 256, H/8, W/8)

        # Bottom-up P3'→P4''
        p3_down_feat = cast(Tensor, self.p3_down(p3_td))  # (B, 128, H/16, W/16)
        p4_bu_cat = lucid.cat([p3_down_feat, p4_lat], dim=1)  # (B, 256, H/16, W/16)
        p4_bu = cast(Tensor, self.p4_csp_bu(p4_bu_cat))  # (B, 256, H/16, W/16)

        # Bottom-up P4''→P5''
        p4_down_feat = cast(Tensor, self.p4_down(p4_bu))  # (B, 256, H/32, W/32)
        p5_bu_cat = lucid.cat([p4_down_feat, p5_lat], dim=1)  # (B, 512, H/32, W/32)
        p5_bu = cast(Tensor, self.p5_csp_bu(p5_bu_cat))  # (B, 512, H/32, W/32)

        return p3_td, p4_bu, p5_bu


# ---------------------------------------------------------------------------
# Box decoding helper (shared with YOLOv3 style)
# ---------------------------------------------------------------------------


def _decode_predictions(
    raw: Tensor,
    anchors_wh: list[tuple[float, float]],
    stride: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode a single-scale raw detection tensor.

    Args:
        raw:         (B, nA*(5+C), H, W)
        anchors_wh:  3 anchor (w, h) pairs for this scale.
        stride:      Feature-map stride.

    Returns:
        (logits, pred_boxes, conf):
          logits    : (B, H*W*nA, C)  raw class logits (pre-sigmoid)
          pred_boxes: (B, H*W*nA, 4)  decoded xyxy pixel boxes
          conf      : (B, H*W*nA)     sigmoid objectness confidence
    """
    B = int(raw.shape[0])
    fH = int(raw.shape[2])
    fW = int(raw.shape[3])
    nA = len(anchors_wh)
    C = int(raw.shape[1]) // nA - 5
    device = raw.device

    raw = raw.reshape(B, nA, 5 + C, fH, fW).permute(0, 1, 3, 4, 2)

    tx = raw[..., 0]
    ty = raw[..., 1]
    tw = raw[..., 2]
    th = raw[..., 3]
    tc = raw[..., 4]
    cls_logits = raw[..., 5:]

    col_data: list[list[float]] = [[float(c) for c in range(fW)] for _ in range(fH)]
    row_data: list[list[float]] = [[float(r)] * fW for r in range(fH)]
    col_t = lucid.tensor(col_data, device=device)
    row_t = lucid.tensor(row_data, device=device)

    px = (F.sigmoid(tx) + col_t) * float(stride)
    py = (F.sigmoid(ty) + row_t) * float(stride)

    aw_data: list[list[list[list[float]]]] = []
    ah_data: list[list[list[list[float]]]] = []
    for _ in range(B):
        b_aw: list[list[list[float]]] = []
        b_ah: list[list[list[float]]] = []
        for a_idx in range(nA):
            aw_val = anchors_wh[a_idx][0]
            ah_val = anchors_wh[a_idx][1]
            b_aw.append([[aw_val] * fW for _ in range(fH)])
            b_ah.append([[ah_val] * fW for _ in range(fH)])
        aw_data.append(b_aw)
        ah_data.append(b_ah)

    aw_t = lucid.tensor(aw_data, device=device)
    ah_t = lucid.tensor(ah_data, device=device)

    pw = lucid.exp(tw) * aw_t
    ph = lucid.exp(th) * ah_t

    x1 = px - pw / 2.0
    y1 = py - ph / 2.0
    x2 = px + pw / 2.0
    y2 = py + ph / 2.0

    boxes = lucid.stack([x1, y1, x2, y2], dim=-1).reshape(B, nA * fH * fW, 4)
    cls_logits = cls_logits.reshape(B, nA * fH * fW, C)
    conf = F.sigmoid(tc.reshape(B, nA * fH * fW))

    return cls_logits, boxes, conf


# ---------------------------------------------------------------------------
# CIoU loss helper
# ---------------------------------------------------------------------------


def _ciou_loss(pred_boxes: Tensor, gt_boxes: Tensor) -> Tensor:
    """Complete IoU loss between paired predicted and GT boxes (xyxy format).

    CIoU = 1 - IoU + d²/c² + α·v
    where:
      d²  = squared Euclidean distance between box centres
      c²  = squared diagonal of the smallest enclosing box
      v   = (4/π²) * (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²
      α   = v / (1 - IoU + v)

    Args:
        pred_boxes: (N, 4) xyxy predicted boxes.
        gt_boxes:   (N, 4) xyxy ground-truth boxes.

    Returns:
        Scalar mean CIoU loss.
    """
    N = int(pred_boxes.shape[0])
    if N == 0:
        return lucid.zeros((1,))

    losses: list[Tensor] = []
    for i in range(N):
        px1 = float(pred_boxes[i, 0].item())
        py1 = float(pred_boxes[i, 1].item())
        px2 = float(pred_boxes[i, 2].item())
        py2 = float(pred_boxes[i, 3].item())
        gx1 = float(gt_boxes[i, 0].item())
        gy1 = float(gt_boxes[i, 1].item())
        gx2 = float(gt_boxes[i, 2].item())
        gy2 = float(gt_boxes[i, 3].item())

        pw = max(px2 - px1, 0.0)
        ph = max(py2 - py1, 0.0)
        gw = max(gx2 - gx1, 0.0)
        gh = max(gy2 - gy1, 0.0)

        pcx = (px1 + px2) / 2.0
        pcy = (py1 + py2) / 2.0
        gcx = (gx1 + gx2) / 2.0
        gcy = (gy1 + gy2) / 2.0

        inter_x1 = max(px1, gx1)
        inter_y1 = max(py1, gy1)
        inter_x2 = min(px2, gx2)
        inter_y2 = min(py2, gy2)
        inter_w = max(inter_x2 - inter_x1, 0.0)
        inter_h = max(inter_y2 - inter_y1, 0.0)
        inter = inter_w * inter_h
        union = pw * ph + gw * gh - inter + 1e-9
        iou = inter / union

        # Smallest enclosing box diagonal²
        enc_x1 = min(px1, gx1)
        enc_y1 = min(py1, gy1)
        enc_x2 = max(px2, gx2)
        enc_y2 = max(py2, gy2)
        enc_w = enc_x2 - enc_x1
        enc_h = enc_y2 - enc_y1
        c_sq = enc_w * enc_w + enc_h * enc_h + 1e-9

        # Centre distance²
        d_sq = (pcx - gcx) ** 2 + (pcy - gcy) ** 2

        # Aspect ratio penalty
        v = (4.0 / (math.pi**2)) * (
            math.atan(gw / (gh + 1e-9)) - math.atan(pw / (ph + 1e-9))
        ) ** 2
        alpha = v / (1.0 - iou + v + 1e-9)

        ciou = iou - d_sq / c_sq - alpha * v
        losses.append(lucid.tensor([[1.0 - ciou]]))

    return lucid.cat(losses).mean()


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def _yolov4_loss(
    raw_preds: list[Tensor],
    targets: list[dict[str, Tensor]],
    config: YOLOV4Config,
) -> Tensor:
    """YOLOv4 multi-scale detection loss with CIoU box regression.

    Args:
        raw_preds: 3 raw tensors (P5, P4, P3) each (B, nA*(5+C), H_l, W_l).
        targets:   List of B target dicts.
        config:    Model configuration.

    Returns:
        Scalar loss tensor.
    """
    B = int(raw_preds[0].shape[0])
    C = config.num_classes
    nA = 3
    anchors_all = config.anchors

    # Scale order: index 0 = P5 (large anchors), 1 = P4, 2 = P3 (small)
    scale_anchor_idx = [(6, 7, 8), (3, 4, 5), (0, 1, 2)]
    strides = [config.strides[2], config.strides[1], config.strides[0]]

    total_loss: list[Tensor] = []

    for raw, anchor_idx_triple, stride in zip(raw_preds, scale_anchor_idx, strides):
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])
        anchors_wh = [anchors_all[i] for i in anchor_idx_triple]

        raw_r = raw.reshape(B, nA, 5 + C, fH, fW).permute(0, 1, 3, 4, 2)

        for b in range(B):
            gt_boxes = targets[b]["boxes"]
            gt_labels = targets[b]["labels"]
            M = int(gt_boxes.shape[0])

            # Build target arrays
            obj_arr: list[list[list[float]]] = [
                [[0.0] * fW for _ in range(fH)] for _ in range(nA)
            ]
            cls_arr: list[list[list[list[float]]]] = [
                [[[0.0] * C for _ in range(fW)] for _ in range(fH)] for _ in range(nA)
            ]
            mask_arr: list[list[list[float]]] = [
                [[0.0] * fW for _ in range(fH)] for _ in range(nA)
            ]

            # Collect positive (pred_box, gt_box) pairs for CIoU
            pos_pred_boxes: list[list[float]] = []
            pos_gt_boxes: list[list[float]] = []

            pred_b = raw_r[b]  # (nA, fH, fW, 5+C)

            if M > 0:
                for m in range(M):
                    x1g = float(gt_boxes[m, 0].item())
                    y1g = float(gt_boxes[m, 1].item())
                    x2g = float(gt_boxes[m, 2].item())
                    y2g = float(gt_boxes[m, 3].item())
                    wg = x2g - x1g
                    hg = y2g - y1g
                    cxg = (x1g + x2g) / 2.0
                    cyg = (y1g + y2g) / 2.0
                    cls_id = int(gt_labels[m].item())

                    col_idx = max(0, min(int(cxg / stride), fW - 1))
                    row_idx = max(0, min(int(cyg / stride), fH - 1))

                    best_iou = -1.0
                    best_a = 0
                    for a_i, (aw, ah) in enumerate(anchors_wh):
                        inter_w = min(wg, aw)
                        inter_h = min(hg, ah)
                        inter = inter_w * inter_h
                        union = wg * hg + aw * ah - inter
                        iou_val = inter / (union + 1e-6)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_a = a_i

                    aw_best, ah_best = anchors_wh[best_a]

                    obj_arr[best_a][row_idx][col_idx] = 1.0
                    mask_arr[best_a][row_idx][col_idx] = 1.0
                    if 0 <= cls_id < C:
                        cls_arr[best_a][row_idx][col_idx][cls_id] = 1.0

                    # Decode predicted box at (best_a, row_idx, col_idx)
                    ptx = float(pred_b[best_a, row_idx, col_idx, 0].item())
                    pty = float(pred_b[best_a, row_idx, col_idx, 1].item())
                    ptw = float(pred_b[best_a, row_idx, col_idx, 2].item())
                    pth = float(pred_b[best_a, row_idx, col_idx, 3].item())

                    pcx = (1.0 / (1.0 + math.exp(-ptx)) + float(col_idx)) * float(
                        stride
                    )
                    pcy = (1.0 / (1.0 + math.exp(-pty)) + float(row_idx)) * float(
                        stride
                    )
                    pw = math.exp(ptw) * aw_best
                    ph = math.exp(pth) * ah_best

                    pos_pred_boxes.append(
                        [pcx - pw / 2.0, pcy - ph / 2.0, pcx + pw / 2.0, pcy + ph / 2.0]
                    )
                    pos_gt_boxes.append([x1g, y1g, x2g, y2g])

            tgt_obj = lucid.tensor(obj_arr)
            tgt_cls = lucid.tensor(cls_arr)
            obj_mask = lucid.tensor(mask_arr)

            pred_tc = pred_b[..., 4]  # (nA, fH, fW)
            pred_cls_logits = pred_b[..., 5:]  # (nA, fH, fW, C)

            # Objectness BCE
            obj_bce = F.binary_cross_entropy_with_logits(
                pred_tc, tgt_obj, reduction="none"
            )
            noobj_mask = 1.0 - obj_mask
            obj_loss = (
                obj_bce * obj_mask + obj_bce * noobj_mask * config.lambda_noobj
            ).sum()

            # Class BCE
            cls_bce = F.binary_cross_entropy_with_logits(
                pred_cls_logits, tgt_cls, reduction="none"
            )
            cls_loss = (cls_bce * obj_mask[..., None]).sum()

            # CIoU loss for positive anchors
            if pos_pred_boxes:
                p_boxes = lucid.tensor(pos_pred_boxes)  # (P, 4)
                g_boxes = lucid.tensor(pos_gt_boxes)  # (P, 4)
                ciou_l = _ciou_loss(p_boxes, g_boxes)
            else:
                ciou_l = lucid.zeros((1,))

            scale_loss = ciou_l.sum() + obj_loss + cls_loss
            total_loss.append(scale_loss.reshape(1))

    if not total_loss:
        return lucid.zeros((1,))
    return lucid.cat(total_loss).sum()


# ---------------------------------------------------------------------------
# YOLOv4 model
# ---------------------------------------------------------------------------


class YOLOV4ForObjectDetection(PretrainedModel):
    """YOLOv4 object detector (Bochkovskiy et al., 2020).

    Input contract
    --------------
    ``x``       : (B, C, H, W) image batch.
    ``targets`` : optional list of B dicts with:
                    ``"boxes"``  — (M_i, 4) xyxy pixel-coordinate boxes.
                    ``"labels"`` — (M_i,)   integer class ids (0-indexed).

    Output contract
    ---------------
    ``ObjectDetectionOutput``:
      ``logits``    : (B, total_anchors, num_classes) raw sigmoid class logits.
      ``pred_boxes``: (B, total_anchors, 4) decoded xyxy pixel boxes.
      ``loss``      : CIoU + BCE loss scalar when targets provided.
    """

    config_class: ClassVar[type[YOLOV4Config]] = YOLOV4Config
    base_model_prefix: ClassVar[str] = "yolo_v4"

    def __init__(self, config: YOLOV4Config) -> None:
        super().__init__(config)
        self._cfg = config
        C = config.num_classes
        nA = 3

        # CSPDarknet-53 backbone
        self.backbone = _CSPDarknet53(config.in_channels)

        # PANet neck
        self.neck = _PANetNeck()

        # Detection heads:
        # P3'' output = 256ch (from p3_td in neck)
        # P4'' output = 256ch (from p4_bu in neck)
        # P5'' output = 512ch (from p5_bu in neck)
        self.p3_head = nn.Sequential(
            _ConvBnLeaky(256, 128, 1),
            _ConvBnLeaky(128, 256, 3),
            nn.Conv2d(256, nA * (5 + C), 1, bias=True),
        )
        self.p4_head = nn.Sequential(
            _ConvBnLeaky(256, 256, 1),
            _ConvBnLeaky(256, 512, 3),
            nn.Conv2d(512, nA * (5 + C), 1, bias=True),
        )
        self.p5_head = nn.Sequential(
            _ConvBnLeaky(512, 256, 1),
            _ConvBnLeaky(256, 512, 3),
            nn.Conv2d(512, nA * (5 + C), 1, bias=True),
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run YOLOv4.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional list of target dicts per image.

        Returns:
            ``ObjectDetectionOutput``:
              ``logits``    : (B, total_anchors, C) raw class logits.
              ``pred_boxes``: (B, total_anchors, 4) xyxy decoded boxes.
              ``loss``      : loss scalar when targets provided.
        """
        cfg = self._cfg

        # Backbone
        p3, p4, p5 = self.backbone.forward(x)

        # PANet neck
        p3_out, p4_out, p5_out = self.neck.forward(p3, p4, p5)

        # Detection heads (P5→large, P4→medium, P3→small)
        p5_raw = cast(Tensor, self.p5_head(p5_out))  # (B, nA*(5+C), H/32, W/32)
        p4_raw = cast(Tensor, self.p4_head(p4_out))  # (B, nA*(5+C), H/16, W/16)
        p3_raw = cast(Tensor, self.p3_head(p3_out))  # (B, nA*(5+C), H/8,  W/8)

        # raw_preds order: P5, P4, P3 (coarse→fine)
        raw_preds = [p5_raw, p4_raw, p3_raw]

        anchors_all = cfg.anchors
        scale_anchors = [
            [anchors_all[6], anchors_all[7], anchors_all[8]],
            [anchors_all[3], anchors_all[4], anchors_all[5]],
            [anchors_all[0], anchors_all[1], anchors_all[2]],
        ]
        scale_strides = [cfg.strides[2], cfg.strides[1], cfg.strides[0]]

        all_logits: list[Tensor] = []
        all_boxes: list[Tensor] = []

        for raw_pred, anch_wh, stride in zip(raw_preds, scale_anchors, scale_strides):
            logits_s, boxes_s, _ = _decode_predictions(raw_pred, anch_wh, stride)
            all_logits.append(logits_s)
            all_boxes.append(boxes_s)

        logits = lucid.cat(all_logits, dim=1)
        pred_boxes = lucid.cat(all_boxes, dim=1)

        loss: Tensor | None = None
        if targets is not None:
            loss = _yolov4_loss(raw_preds, targets, cfg)

        return ObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            loss=loss,
        )

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_sizes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """Filter by score, clip boxes, apply per-class NMS.

        Args:
            output:      Forward pass output.
            image_sizes: List of (H, W) per image.

        Returns:
            Per-image list of dicts with "boxes", "scores", "labels".
        """
        B = int(output.logits.shape[0])
        results: list[dict[str, Tensor]] = []

        for b in range(B):
            cls_logits = output.logits[b]
            boxes = output.pred_boxes[b]
            iH, iW = image_sizes[b]

            cls_probs = F.sigmoid(cls_logits)
            N_anc = int(cls_probs.shape[0])
            C = int(cls_probs.shape[1])

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for a in range(N_anc):
                for c in range(C):
                    sc = float(cls_probs[a, c].item())
                    if sc >= self._cfg.score_thresh:
                        keep_boxes.append(boxes[a : a + 1])
                        keep_scores.append(lucid.tensor([[sc]]))
                        keep_labels.append(lucid.tensor([[float(c)]]))

            if not keep_boxes:
                results.append(
                    {
                        "boxes": lucid.zeros((0, 4)),
                        "scores": lucid.zeros((0,)),
                        "labels": lucid.zeros((0,)),
                    }
                )
                continue

            det_boxes = lucid.cat(keep_boxes, dim=0)
            det_scores = lucid.cat(keep_scores, dim=0).reshape(-1)
            det_labels = lucid.cat(keep_labels, dim=0).reshape(-1)

            det_boxes = clip_boxes_to_image(det_boxes, (iH, iW))

            keep_idx = batched_nms(
                det_boxes, det_scores, det_labels, self._cfg.nms_thresh
            )
            K2 = int(keep_idx.shape[0])
            if K2 == 0:
                results.append(
                    {
                        "boxes": lucid.zeros((0, 4)),
                        "scores": lucid.zeros((0,)),
                        "labels": lucid.zeros((0,)),
                    }
                )
                continue

            idx_list: list[int] = [int(keep_idx[i].item()) for i in range(K2)]
            idx_t = lucid.tensor(idx_list)
            results.append(
                {
                    "boxes": det_boxes[idx_t],
                    "scores": det_scores[idx_t],
                    "labels": det_labels[idx_t],
                }
            )

        return results


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v4",
    model_class=YOLOV4ForObjectDetection,
)
def yolo_v4(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOV4ForObjectDetection:
    """YOLOv4 with CSPDarknet-53 + SPP + PANet, 3-scale detection (COCO 80 classes).

    Reference: Bochkovskiy et al., "YOLOv4: Optimal Speed and Accuracy of
    Object Detection", arXiv 2020.

    Args:
        pretrained: If True, attempt to load pretrained COCO weights.
        **overrides: Override any ``YOLOV4Config`` field.

    Returns:
        ``YOLOV4ForObjectDetection`` instance.
    """
    config = YOLOV4Config(**{k: v for k, v in overrides.items()})  # type: ignore[arg-type]
    model = YOLOV4ForObjectDetection(config)
    if pretrained:
        raise NotImplementedError("Pretrained YOLOv4 weights are not yet available.")
    return model
