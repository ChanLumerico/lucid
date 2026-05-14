"""YOLOv3 — You Only Look Once v3 (Redmon & Farhadi, arXiv 2018).

Paper: "YOLOv3: An Incremental Improvement"
https://arxiv.org/abs/1804.02767

Key design
----------
YOLOv3 extends YOLO by:
  1. Darknet-53 backbone — 53-layer convolutional network with residual
     connections and no pooling (stride-2 convolutions for downsampling).
  2. Multi-scale detection at 3 granularities (P3/P4/P5 — strides 8/16/32),
     each with 3 anchors, enabling small/medium/large object detection.
  3. Independent sigmoid activations per class (multi-label) rather than
     softmax, making class predictions class-independent.

Architecture overview
---------------------
  Image (B, 3, H, W)
    ↓  Darknet-53 backbone (stages 1–5)
       - Stem:  Conv(3→32, k3) → Conv(32→64, k3, stride=2)
       - Stage1: 1×dark53_block(64→64)    → stride-2 conv → 128ch
       - Stage2: 2×dark53_block(128→128)  → stride-2 conv → 256ch  [P3_raw]
       - Stage3: 8×dark53_block(256→256)  → stride-2 conv → 512ch  [P4_raw]
       - Stage4: 8×dark53_block(512→512)  → stride-2 conv → 1024ch
       - Stage5: 4×dark53_block(1024→1024) [P5]
    ↓  Detection heads at P5, P4, P3
       - P5:  5×Conv(1×1/3×3 alternating) → 1×Conv → 3*(5+C) predictions
       - P4:  P5_feature upsample+concat(P4_raw) → same head sequence
       - P3:  P4_feature upsample+concat(P3_raw) → same head sequence
    ↓  Decode each scale: sigmoid(tx,ty)+offset, exp(tw,th)*anchor, sigmoid(obj), sigmoid(cls)
    ↓  ObjectDetectionOutput(logits, pred_boxes, loss)

Loss (training)
---------------
  Requires ``targets`` — list of B dicts:
    "boxes"  : (M, 4) xyxy pixel coordinates
    "labels" : (M,)   integer class ids (0-indexed)

  Assignment: for each GT box, the anchor with highest IoU at the matching
  grid cell is made a positive. All other anchors are negatives.
  L = lambda_coord * L_mse(box)   [positive anchors only]
    + L_bce(obj)                  [positive=1, negative=lambda_noobj scaled]
    + L_bce(cls)                  [positive anchors only]
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
class YOLOV3Config(ModelConfig):
    """Configuration for YOLOv3.

    YOLOv3 (Redmon & Farhadi, 2018) uses a Darknet-53 backbone with three
    detection scales at strides 8, 16, and 32 (P3/P4/P5).  Each scale has 3
    hand-picked COCO anchors for small, medium, and large objects respectively.

    Args:
        num_classes:   Number of foreground classes (COCO default = 80).
        in_channels:   Input image channels.
        anchors:       9 (width, height) anchor pairs in pixels, 3 per scale.
                       Order: P3 (small), P4 (medium), P5 (large).
        strides:       Feature-map strides for P3, P4, P5.
        score_thresh:  Minimum objectness × class score to keep a detection.
        nms_thresh:    IoU threshold for per-class NMS.
        lambda_coord:  Box regression loss weight.
        lambda_noobj:  Objectness loss weight for negative anchors.
    """

    model_type: ClassVar[str] = "yolo_v3"

    num_classes: int = 80
    in_channels: int = 3

    # 9 COCO anchor (w, h) pairs — 3 per detection scale
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
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "anchors", tuple(tuple(a) for a in self.anchors))
        object.__setattr__(self, "strides", tuple(self.strides))


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _ConvBnLeaky(nn.Module):
    """Conv2d(bias=False) → BatchNorm2d → LeakyReLU(0.1)."""

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


class _Dark53Block(nn.Module):
    """Darknet-53 residual unit: Conv(1×1) → Conv(3×3) + skip."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        mid_ch = in_ch // 2
        self.conv1 = _ConvBnLeaky(in_ch, mid_ch, 1)
        self.conv2 = _ConvBnLeaky(mid_ch, in_ch, 3)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return x + cast(Tensor, self.conv2(cast(Tensor, self.conv1(x))))


def _make_dark53_stage(
    in_ch: int, out_ch: int, n_blocks: int
) -> tuple[nn.Sequential, nn.Module]:
    """Build one Darknet-53 stage.

    Returns:
        (blocks, downsample_conv) — residual blocks and the stride-2 conv that
        follows them to double channels and halve spatial resolution.
    """
    blocks: list[nn.Module] = [_Dark53Block(in_ch) for _ in range(n_blocks)]
    stage = nn.Sequential(*blocks)
    down = _ConvBnLeaky(in_ch, out_ch, 3, stride=2)
    return stage, down


# ---------------------------------------------------------------------------
# Darknet-53 backbone
# ---------------------------------------------------------------------------


class _Darknet53(nn.Module):
    """Darknet-53 backbone.

    Returns three feature maps for multi-scale detection:
      P3_raw : (B, 256, H/8,  W/8)
      P4_raw : (B, 512, H/16, W/16)
      P5     : (B, 1024, H/32, W/32)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stem
        self.conv0 = _ConvBnLeaky(in_channels, 32, 3)
        self.conv1 = _ConvBnLeaky(32, 64, 3, stride=2)

        # Stage 1: 1×block(64), then stride-2 → 128ch
        self.stage1_blocks, self.down1 = _make_dark53_stage(64, 128, 1)

        # Stage 2: 2×block(128), then stride-2 → 256ch  [P3_raw]
        self.stage2_blocks, self.down2 = _make_dark53_stage(128, 256, 2)

        # Stage 3: 8×block(256), then stride-2 → 512ch  [P4_raw]
        self.stage3_blocks, self.down3 = _make_dark53_stage(256, 512, 8)

        # Stage 4: 8×block(512), then stride-2 → 1024ch
        self.stage4_blocks, self.down4 = _make_dark53_stage(512, 1024, 8)

        # Stage 5: 4×block(1024) [P5 final]
        self.stage5_blocks = nn.Sequential(*[_Dark53Block(1024) for _ in range(4)])

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:  # type: ignore[override]
        x = cast(Tensor, self.conv1(cast(Tensor, self.conv0(x))))

        x = cast(Tensor, self.stage1_blocks(x))
        x = cast(Tensor, self.down1(x))

        x = cast(Tensor, self.stage2_blocks(x))
        p3_raw = x  # (B, 256, H/8, W/8)
        x = cast(Tensor, self.down2(x))

        x = cast(Tensor, self.stage3_blocks(x))
        p4_raw = x  # (B, 512, H/16, W/16)
        x = cast(Tensor, self.down3(x))

        x = cast(Tensor, self.stage4_blocks(x))
        x = cast(Tensor, self.down4(x))

        p5 = cast(Tensor, self.stage5_blocks(x))  # (B, 1024, H/32, W/32)

        return p3_raw, p4_raw, p5


# ---------------------------------------------------------------------------
# Tiny backbone (YOLOv3-Tiny)
# ---------------------------------------------------------------------------


class _Darknet53Tiny(nn.Module):
    """Lightweight Darknet backbone used by YOLOv3-Tiny.

    6 conv stages with MaxPool (no residuals), returns two feature maps:
      P4_raw : (B, 256, H/16, W/16)
      P5     : (B, 1024, H/32, W/32)

    Only 2 detection scales are used in the Tiny variant.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv1 = _ConvBnLeaky(in_channels, 16, 3)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = _ConvBnLeaky(16, 32, 3)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = _ConvBnLeaky(32, 64, 3)
        self.pool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = _ConvBnLeaky(64, 128, 3)
        self.pool4 = nn.MaxPool2d(2, stride=2)
        self.conv5 = _ConvBnLeaky(128, 256, 3)  # P4_raw
        self.pool5 = nn.MaxPool2d(2, stride=2)
        self.conv6 = _ConvBnLeaky(256, 512, 3)
        # Stride-1 MaxPool with same-pad to keep spatial size
        self.pool6 = nn.MaxPool2d(2, stride=1, padding=1)
        self.conv7 = _ConvBnLeaky(512, 1024, 3)
        self.conv8 = _ConvBnLeaky(1024, 256, 1)  # bottleneck before P5 head

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        x = cast(Tensor, self.pool1(cast(Tensor, self.conv1(x))))
        x = cast(Tensor, self.pool2(cast(Tensor, self.conv2(x))))
        x = cast(Tensor, self.pool3(cast(Tensor, self.conv3(x))))
        x = cast(Tensor, self.pool4(cast(Tensor, self.conv4(x))))
        x = cast(Tensor, self.conv5(x))
        p4_raw = x  # (B, 256, H/16, W/16)
        x = cast(Tensor, self.pool5(x))
        x = cast(Tensor, self.pool6(cast(Tensor, self.conv6(x))))
        x = cast(Tensor, self.conv7(x))
        p5 = cast(Tensor, self.conv8(x))  # (B, 256, H/32, W/32)
        return p4_raw, p5


# ---------------------------------------------------------------------------
# Detection head
# ---------------------------------------------------------------------------


def _make_detection_head(
    in_ch: int, mid_ch: int, num_anchors: int, num_classes: int
) -> tuple[nn.Sequential, nn.Conv2d]:
    """Build a YOLOv3 detection head.

    5 alternating 1×1/3×3 convolutions for feature compression, followed by a
    final 1×1 conv that outputs ``num_anchors*(5+num_classes)`` channels.

    Returns:
        (compress_convs, predict_conv)
    """
    c = [
        _ConvBnLeaky(in_ch, mid_ch, 1),
        _ConvBnLeaky(mid_ch, mid_ch * 2, 3),
        _ConvBnLeaky(mid_ch * 2, mid_ch, 1),
        _ConvBnLeaky(mid_ch, mid_ch * 2, 3),
        _ConvBnLeaky(mid_ch * 2, mid_ch, 1),
    ]
    compress = nn.Sequential(*c)
    predict = nn.Conv2d(mid_ch, num_anchors * (5 + num_classes), 1, bias=True)
    return compress, predict


# ---------------------------------------------------------------------------
# Box decoding helper
# ---------------------------------------------------------------------------


def _decode_predictions(
    raw: Tensor,
    anchors_wh: list[tuple[float, float]],
    stride: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Decode a single-scale raw detection tensor.

    Args:
        raw:         (B, num_anchors*(5+C), H, W) raw network output.
        anchors_wh:  3 anchor (w, h) pairs in pixels for this scale.
        stride:      Stride of this feature map.

    Returns:
        (logits, pred_boxes, conf):
          logits    : (B, H*W*3, C)   raw (pre-sigmoid) class logits
          pred_boxes: (B, H*W*3, 4)   decoded xyxy pixel boxes
          conf      : (B, H*W*3)      objectness confidence (sigmoid)
    """
    B = int(raw.shape[0])
    fH = int(raw.shape[2])
    fW = int(raw.shape[3])
    nA = len(anchors_wh)
    C = int(raw.shape[1]) // nA - 5
    device = raw.device

    # Reshape → (B, nA, 5+C, H, W) → (B, nA, H, W, 5+C)
    raw = raw.reshape(B, nA, 5 + C, fH, fW)
    raw = raw.permute(0, 1, 3, 4, 2)  # (B, nA, fH, fW, 5+C)

    tx = raw[..., 0]  # (B, nA, fH, fW)
    ty = raw[..., 1]
    tw = raw[..., 2]
    th = raw[..., 3]
    tc = raw[..., 4]  # objectness logit
    cls_logits = raw[..., 5:]  # (B, nA, fH, fW, C)

    # Build cell offset grids as Python-level lists → tensor
    col_data: list[list[float]] = [[float(c) for c in range(fW)] for _ in range(fH)]
    row_data: list[list[float]] = [[float(r)] * fW for r in range(fH)]

    col_t = lucid.tensor(col_data, device=device)  # (fH, fW)
    row_t = lucid.tensor(row_data, device=device)  # (fH, fW)

    # Decode box centres
    px = (F.sigmoid(tx) + col_t) * float(stride)  # (B, nA, fH, fW)
    py = (F.sigmoid(ty) + row_t) * float(stride)

    # Build anchor arrays as Python lists → tensors
    aw_data: list[list[list[list[float]]]] = []
    ah_data: list[list[list[list[float]]]] = []
    for _ in range(B):
        b_aw: list[list[list[float]]] = []
        b_ah: list[list[list[float]]] = []
        for a_idx in range(nA):
            aw_val = anchors_wh[a_idx][0]
            ah_val = anchors_wh[a_idx][1]
            row_aw: list[list[float]] = [[aw_val] * fW for _ in range(fH)]
            row_ah: list[list[float]] = [[ah_val] * fW for _ in range(fH)]
            b_aw.append(row_aw)
            b_ah.append(row_ah)
        aw_data.append(b_aw)
        ah_data.append(b_ah)

    aw_t = lucid.tensor(aw_data, device=device)  # (B, nA, fH, fW)
    ah_t = lucid.tensor(ah_data, device=device)

    pw = lucid.exp(tw) * aw_t  # (B, nA, fH, fW)
    ph = lucid.exp(th) * ah_t

    # xyxy boxes in pixel space
    x1 = px - pw / 2.0
    y1 = py - ph / 2.0
    x2 = px + pw / 2.0
    y2 = py + ph / 2.0

    boxes = lucid.stack([x1, y1, x2, y2], dim=-1)  # (B, nA, fH, fW, 4)

    N = nA * fH * fW
    boxes = boxes.reshape(B, N, 4)
    cls_logits = cls_logits.reshape(B, N, C)
    conf = F.sigmoid(tc.reshape(B, N))

    return cls_logits, boxes, conf


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------


def _yolov3_loss(
    raw_preds: list[Tensor],
    targets: list[dict[str, Tensor]],
    config: YOLOV3Config,
) -> Tensor:
    """Compute YOLOv3 multi-scale detection loss.

    Args:
        raw_preds: 3 raw tensors per scale (P5, P4, P3), each
                   (B, nA*(5+C), H_l, W_l).
        targets:   List of B target dicts (boxes xyxy pixels, labels ints).
        config:    Model configuration.

    Returns:
        Scalar loss tensor.
    """
    B = int(raw_preds[0].shape[0])
    C = config.num_classes
    nA = 3  # anchors per scale
    anchors_all = config.anchors  # 9 anchor pairs

    # Map scale index → anchor subset and stride
    # scale 0 = P5 (large), scale 1 = P4 (medium), scale 2 = P3 (small)
    # Anchors are stored small→large: indices 0-2 = P3, 3-5 = P4, 6-8 = P5
    scale_anchor_idx = [(6, 7, 8), (3, 4, 5), (0, 1, 2)]
    strides = [config.strides[2], config.strides[1], config.strides[0]]

    total_loss: list[Tensor] = []

    for scale_i, (raw, anchor_idx_triple, stride) in enumerate(
        zip(raw_preds, scale_anchor_idx, strides)
    ):
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])
        anchors_wh = [anchors_all[i] for i in anchor_idx_triple]

        # Reshape raw → (B, nA, 5+C, fH, fW) → (B, nA, fH, fW, 5+C)
        raw_r = raw.reshape(B, nA, 5 + C, fH, fW).permute(0, 1, 3, 4, 2)

        for b in range(B):
            gt_boxes = targets[b]["boxes"]  # (M, 4) xyxy pixels
            gt_labels = targets[b]["labels"]  # (M,)
            M = int(gt_boxes.shape[0])

            # Build target tensors (floats) for this image and scale
            tgt_tx = lucid.zeros((nA, fH, fW))
            tgt_ty = lucid.zeros((nA, fH, fW))
            tgt_tw = lucid.zeros((nA, fH, fW))
            tgt_th = lucid.zeros((nA, fH, fW))
            tgt_obj = lucid.zeros((nA, fH, fW))
            tgt_cls = lucid.zeros((nA, fH, fW, C))
            obj_mask = lucid.zeros((nA, fH, fW))  # 1 for positives

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

                    # Grid cell
                    col_idx = int(cxg / stride)
                    row_idx = int(cyg / stride)
                    col_idx = max(0, min(col_idx, fW - 1))
                    row_idx = max(0, min(row_idx, fH - 1))

                    # Find best anchor (centred at 0 IoU match)
                    best_iou = -1.0
                    best_a = 0
                    for a_i, (aw, ah) in enumerate(anchors_wh):
                        # IoU between GT box (centred at origin) and anchor
                        inter_w = min(wg, aw)
                        inter_h = min(hg, ah)
                        inter = inter_w * inter_h
                        union = wg * hg + aw * ah - inter
                        iou_val = inter / (union + 1e-6)
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_a = a_i

                    aw_best, ah_best = anchors_wh[best_a]

                    # Regression targets in grid-cell space
                    tx_tgt = cxg / stride - float(col_idx)
                    ty_tgt = cyg / stride - float(row_idx)
                    tw_tgt = math.log(max(wg, 1.0) / aw_best + 1e-9)
                    th_tgt = math.log(max(hg, 1.0) / ah_best + 1e-9)

                    # Write targets using Python-level scalar assignment via
                    # list construction + lucid.tensor (no in-place indexing needed)
                    # We accumulate into lists, then form full tensors at the end.
                    # For simplicity, we build mutable Python arrays and convert once.
                    # We track positives with a list below.
                    _ = (tx_tgt, ty_tgt, tw_tgt, th_tgt)  # computed but used below

                    # Mark positive
                    # Build coordinate index to update (a, r, c) positions
                    # We must do this via slice assignment using tensor indexing
                    # (Lucid supports in-place via __setitem__ if available, or
                    #  we collect then construct).
                    # Strategy: accumulate Python-list targets, then build tensor.
                    pass

                # Rebuild entire target arrays from scratch using Python lists.
                # This avoids any in-place mutation on Lucid tensors.
                tx_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                ty_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                tw_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                th_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                obj_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]
                cls_arr: list[list[list[list[float]]]] = [
                    [[[0.0] * C for _ in range(fW)] for _ in range(fH)]
                    for _ in range(nA)
                ]
                mask_arr: list[list[list[float]]] = [
                    [[0.0] * fW for _ in range(fH)] for _ in range(nA)
                ]

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
                    tx_arr[best_a][row_idx][col_idx] = cxg / stride - float(col_idx)
                    ty_arr[best_a][row_idx][col_idx] = cyg / stride - float(row_idx)
                    tw_arr[best_a][row_idx][col_idx] = math.log(
                        max(wg, 1.0) / aw_best + 1e-9
                    )
                    th_arr[best_a][row_idx][col_idx] = math.log(
                        max(hg, 1.0) / ah_best + 1e-9
                    )
                    obj_arr[best_a][row_idx][col_idx] = 1.0
                    mask_arr[best_a][row_idx][col_idx] = 1.0
                    if 0 <= cls_id < C:
                        cls_arr[best_a][row_idx][col_idx][cls_id] = 1.0

                tgt_tx = lucid.tensor(tx_arr)
                tgt_ty = lucid.tensor(ty_arr)
                tgt_tw = lucid.tensor(tw_arr)
                tgt_th = lucid.tensor(th_arr)
                tgt_obj = lucid.tensor(obj_arr)
                tgt_cls = lucid.tensor(cls_arr)
                obj_mask = lucid.tensor(mask_arr)

            # Predicted values for this image
            pred_b = raw_r[b]  # (nA, fH, fW, 5+C)
            pred_tx = pred_b[..., 0]
            pred_ty = pred_b[..., 1]
            pred_tw = pred_b[..., 2]
            pred_th = pred_b[..., 3]
            pred_tc = pred_b[..., 4]
            pred_cls_logits = pred_b[..., 5:]  # (nA, fH, fW, C)

            # Box MSE loss (positive anchors only)
            box_loss_sx = ((F.sigmoid(pred_tx) - tgt_tx) ** 2) * obj_mask
            box_loss_sy = ((F.sigmoid(pred_ty) - tgt_ty) ** 2) * obj_mask
            box_loss_w = ((pred_tw - tgt_tw) ** 2) * obj_mask
            box_loss_h = ((pred_th - tgt_th) ** 2) * obj_mask
            box_loss = (box_loss_sx + box_loss_sy + box_loss_w + box_loss_h).sum()

            # Objectness BCE
            obj_bce = F.binary_cross_entropy_with_logits(
                pred_tc, tgt_obj, reduction="none"
            )
            # negative anchors get lambda_noobj weight
            noobj_mask = 1.0 - obj_mask
            obj_loss = (
                obj_bce * obj_mask + obj_bce * noobj_mask * config.lambda_noobj
            ).sum()

            # Class BCE (positive anchors only)
            cls_bce = F.binary_cross_entropy_with_logits(
                pred_cls_logits, tgt_cls, reduction="none"
            )
            cls_loss = (cls_bce * obj_mask[..., None]).sum()

            scale_loss = config.lambda_coord * box_loss + obj_loss + cls_loss
            total_loss.append(scale_loss.reshape(1))

    if not total_loss:
        return lucid.zeros((1,))
    return lucid.cat(total_loss).sum()


# ---------------------------------------------------------------------------
# YOLOv3 model
# ---------------------------------------------------------------------------


class YOLOV3ForObjectDetection(PretrainedModel):
    """YOLOv3 object detector (Redmon & Farhadi, 2018).

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
      ``loss``      : detection loss scalar when targets provided.
    """

    config_class: ClassVar[type[YOLOV3Config]] = YOLOV3Config
    base_model_prefix: ClassVar[str] = "yolo_v3"

    def __init__(self, config: YOLOV3Config) -> None:
        super().__init__(config)
        self._cfg = config
        C = config.num_classes
        nA = 3  # anchors per scale

        # Backbone
        self.backbone = _Darknet53(config.in_channels)

        # Actual backbone output channels:
        #   p3_raw : 128ch  (stride-8  feature, before down2)
        #   p4_raw : 256ch  (stride-16 feature, before down3)
        #   p5     : 1024ch (stride-32 final stage)

        # P5 head (1024ch input): compress mid_ch=512
        self.p5_compress, self.p5_predict = _make_detection_head(1024, 512, nA, C)
        # p5_compress output: 512ch

        # Upsample + concat: p5_up (256ch) + p4_raw (256ch) = 512ch
        self.p5_to_p4_conv = _ConvBnLeaky(512, 256, 1)  # 512→256
        self.p4_compress, self.p4_predict = _make_detection_head(256 + 256, 256, nA, C)
        # p4_compress output: 256ch

        # Upsample + concat: p4_up (128ch) + p3_raw (128ch) = 256ch
        self.p4_to_p3_conv = _ConvBnLeaky(256, 128, 1)  # 256→128
        self.p3_compress, self.p3_predict = _make_detection_head(128 + 128, 128, nA, C)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        """Run YOLOv3.

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
        B = int(x.shape[0])

        # Backbone
        p3_raw, p4_raw, p5 = self.backbone.forward(x)

        # P5 branch
        p5_feat = cast(Tensor, self.p5_compress(p5))  # (B, 512, H/32, W/32)
        p5_raw = cast(Tensor, self.p5_predict(p5_feat))  # (B, nA*(5+C), H/32, W/32)

        # P4 branch: upsample P5 → concat with P4_raw
        p5_up = cast(Tensor, self.p5_to_p4_conv(p5_feat))  # (B, 256, H/32, W/32)
        fH4 = int(p4_raw.shape[2])
        fW4 = int(p4_raw.shape[3])
        p5_up = F.interpolate(p5_up, size=(fH4, fW4), mode="nearest")
        p4_cat = lucid.cat([p5_up, p4_raw], dim=1)  # (B, 256+512, H/16, W/16)
        p4_feat = cast(Tensor, self.p4_compress(p4_cat))  # (B, 256, H/16, W/16)
        p4_raw_pred = cast(
            Tensor, self.p4_predict(p4_feat)
        )  # (B, nA*(5+C), H/16, W/16)

        # P3 branch: upsample P4 → concat with P3_raw
        p4_up = cast(Tensor, self.p4_to_p3_conv(p4_feat))  # (B, 128, H/16, W/16)
        fH3 = int(p3_raw.shape[2])
        fW3 = int(p3_raw.shape[3])
        p4_up = F.interpolate(p4_up, size=(fH3, fW3), mode="nearest")
        p3_cat = lucid.cat([p4_up, p3_raw], dim=1)  # (B, 128+256, H/8, W/8)
        p3_feat = cast(Tensor, self.p3_compress(p3_cat))  # (B, 128, H/8, W/8)
        p3_raw_pred = cast(Tensor, self.p3_predict(p3_feat))  # (B, nA*(5+C), H/8, W/8)

        # raw_preds order: P5, P4, P3 (from coarse to fine)
        raw_preds = [p5_raw, p4_raw_pred, p3_raw_pred]

        # Anchor subsets: scale 0=P5(large), 1=P4(medium), 2=P3(small)
        anchors_all = cfg.anchors
        scale_anchors = [
            [anchors_all[6], anchors_all[7], anchors_all[8]],
            [anchors_all[3], anchors_all[4], anchors_all[5]],
            [anchors_all[0], anchors_all[1], anchors_all[2]],
        ]
        scale_strides = [cfg.strides[2], cfg.strides[1], cfg.strides[0]]

        # Decode all scales
        all_logits: list[Tensor] = []
        all_boxes: list[Tensor] = []

        for raw_pred, anch_wh, stride in zip(raw_preds, scale_anchors, scale_strides):
            logits_s, boxes_s, _ = _decode_predictions(raw_pred, anch_wh, stride)
            all_logits.append(logits_s)
            all_boxes.append(boxes_s)

        logits = lucid.cat(all_logits, dim=1)  # (B, total_anchors, C)
        pred_boxes = lucid.cat(all_boxes, dim=1)  # (B, total_anchors, 4)

        # Loss
        loss: Tensor | None = None
        if targets is not None:
            loss = _yolov3_loss(raw_preds, targets, cfg)

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
            cls_logits = output.logits[b]  # (total_anchors, C)
            boxes = output.pred_boxes[b]  # (total_anchors, 4)
            iH, iW = image_sizes[b]

            cls_probs = F.sigmoid(cls_logits)  # (total_anchors, C)
            N_anc = int(cls_probs.shape[0])
            C = int(cls_probs.shape[1])

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for a in range(N_anc):
                for c in range(C):
                    sc = float(cls_probs[a, c].item())
                    if sc >= self._cfg.score_thresh:
                        keep_boxes.append(boxes[a : a + 1])  # (1, 4)
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

            det_boxes = lucid.cat(keep_boxes, dim=0)  # (K, 4)
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
# YOLOv3-Tiny model
# ---------------------------------------------------------------------------


class _YOLOV3Tiny(PretrainedModel):
    """YOLOv3-Tiny — 2-scale lightweight variant.

    Uses _Darknet53Tiny backbone (6 conv stages + maxpool, no residuals)
    and detects at P5 (stride 32) and P4 (stride 16) only.
    """

    config_class: ClassVar[type[YOLOV3Config]] = YOLOV3Config
    base_model_prefix: ClassVar[str] = "yolo_v3_tiny"

    def __init__(self, config: YOLOV3Config) -> None:
        super().__init__(config)
        self._cfg = config
        C = config.num_classes
        nA = 3

        self.backbone = _Darknet53Tiny(config.in_channels)

        # P5 head: tiny backbone outputs 256ch at P5
        self.p5_compress, self.p5_predict = _make_detection_head(256, 128, nA, C)

        # P4 branch: upsample P5 (128ch) + concat P4_raw (256ch) → 384ch
        self.p5_to_p4_conv = _ConvBnLeaky(128, 128, 1)
        self.p4_compress, self.p4_predict = _make_detection_head(256 + 128, 128, nA, C)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
    ) -> ObjectDetectionOutput:
        cfg = self._cfg
        anchors_all = cfg.anchors

        p4_raw, p5 = self.backbone.forward(x)

        p5_feat = cast(Tensor, self.p5_compress(p5))
        p5_raw_pred = cast(Tensor, self.p5_predict(p5_feat))

        p5_up = cast(Tensor, self.p5_to_p4_conv(p5_feat))
        fH4 = int(p4_raw.shape[2])
        fW4 = int(p4_raw.shape[3])
        p5_up = F.interpolate(p5_up, size=(fH4, fW4), mode="nearest")
        p4_cat = lucid.cat([p5_up, p4_raw], dim=1)
        p4_feat = cast(Tensor, self.p4_compress(p4_cat))
        p4_raw_pred = cast(Tensor, self.p4_predict(p4_feat))

        raw_preds = [p5_raw_pred, p4_raw_pred]
        scale_anchors = [
            [anchors_all[6], anchors_all[7], anchors_all[8]],
            [anchors_all[3], anchors_all[4], anchors_all[5]],
        ]
        scale_strides = [cfg.strides[2], cfg.strides[1]]

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
            loss = _yolov3_loss(raw_preds, targets, cfg)

        return ObjectDetectionOutput(
            logits=logits,
            pred_boxes=pred_boxes,
            loss=loss,
        )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v3",
    model_class=YOLOV3ForObjectDetection,
)
def yolo_v3(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOV3ForObjectDetection:
    """YOLOv3 with Darknet-53 backbone, 3-scale detection (COCO 80 classes).

    Reference: Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018.

    Args:
        pretrained: If True, attempt to load pretrained COCO weights.
        **overrides: Override any ``YOLOV3Config`` field.

    Returns:
        ``YOLOV3ForObjectDetection`` instance.
    """
    config = YOLOV3Config(**{k: v for k, v in overrides.items()})  # type: ignore[arg-type]
    model = YOLOV3ForObjectDetection(config)
    if pretrained:
        raise NotImplementedError("Pretrained YOLOv3 weights are not yet available.")
    return model


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v3_tiny",
    model_class=_YOLOV3Tiny,
)
def yolo_v3_tiny(
    pretrained: bool = False,
    **overrides: object,
) -> _YOLOV3Tiny:
    """YOLOv3-Tiny — lightweight 2-scale variant with Darknet-Tiny backbone.

    Reference: Redmon & Farhadi, "YOLOv3: An Incremental Improvement", 2018.

    Args:
        pretrained: If True, attempt to load pretrained weights.
        **overrides: Override any ``YOLOV3Config`` field.

    Returns:
        ``_YOLOV3Tiny`` instance.
    """
    config = YOLOV3Config(**{k: v for k, v in overrides.items()})  # type: ignore[arg-type]
    model = _YOLOV3Tiny(config)
    if pretrained:
        raise NotImplementedError(
            "Pretrained YOLOv3-Tiny weights are not yet available."
        )
    return model
