"""YOLOv2 / YOLO9000 — Redmon & Farhadi, CVPR 2017.

Paper: "YOLO9000: Better, Faster, Stronger"

Key innovations over YOLOv1
----------------------------
- **Anchor boxes**: replaces per-cell bounding box predictions with
  dimension-cluster anchors (5 per cell, default COCO clusters).
- **Darknet-19 backbone**: 19 conv layers with batch normalisation and
  no fully connected layers; stride-32 feature map.
- **Passthrough layer**: concatenates the 26×26 (stride-16) feature map
  with the 13×13 (stride-32) map via space-to-depth (2×) to preserve
  fine-grained spatial detail.
- **Direct location prediction**: tx, ty are passed through sigmoid so
  they stay in the cell [0,1]; tw, th remain raw (log-space anchor offsets).
- Multi-scale training (not implemented here — static 416×416).

Architecture (Darknet-19)
-------------------------
  3→32 (3×3) + pool
  32→64 (3×3) + pool
  64→128 (3×3), 128→64 (1×1), 64→128 (3×3) + pool
  128→256 (3×3), 256→128 (1×1), 128→256 (3×3) + pool
  256→512 (3×3), 512→256 (1×1), 256→512 (3×3),
                 512→256 (1×1), 256→512 (3×3)   ← passthrough hook (26×26, 512ch)
  + pool
  512→1024 (3×3), 1024→512 (1×1), 512→1024 (3×3),
                  1024→512 (1×1), 512→1024 (3×3)

Detection head
--------------
  1024→1024 (3×3) × 2
  Passthrough: space_to_depth(26×26, s=2) → 2048ch; cat → 3072ch
  3072→1024 (1×1)
  1024 → num_anchors × (5 + num_classes) (1×1, no BN/act)

Output: (B, A*(5+C), H, W) → (B, H*W*A, 5+C)
  Per anchor: [tx, ty, tw, th, conf, cls_0, …, cls_C-1]
  Decoded:    [sigmoid(tx)+c, sigmoid(ty)+r, pw·exp(tw), ph·exp(th)]
              conf = sigmoid(conf_raw), classes = raw (softmax at inference)

Losses (training)
-----------------
Requires ``targets`` — list of B dicts with:
  ``"boxes"``  : (M_i, 4) xyxy, normalised to [0, 1].
  ``"labels"`` : (M_i,)  integer foreground class ids (0-indexed).

  L = λ_coord · L_xy  +  λ_coord · L_wh  +  L_conf_obj
    +  λ_noobj · L_conf_noobj  +  L_cls

where the responsible anchor is the one with highest IoU vs GT.
"""

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import ModelConfig, PretrainedModel
from lucid.models._output import ObjectDetectionOutput
from lucid.models._registry import register_model
from lucid.models._utils._detection import batched_nms

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default COCO anchor clusters (relative to 32-pixel cells)
_COCO_ANCHORS: tuple[tuple[float, float], ...] = (
    (1.3221, 1.73145),
    (3.19275, 4.00944),
    (5.05587, 8.09892),
    (9.47112, 4.84053),
    (11.2364, 10.0071),
)


@dataclass(frozen=True)
class YOLOV2Config(ModelConfig):
    """Configuration for YOLOv2 / YOLO9000 (Redmon & Farhadi, CVPR 2017).

    Args:
        num_classes:  Number of object classes C (COCO default: 80).
        in_channels:  Input image channels (default 3).
        num_anchors:  Number of anchor boxes per cell A (default 5).
        anchors:      Anchor (width, height) pairs relative to a 32-pixel cell.
                      Must have exactly ``num_anchors`` entries.
        score_thresh: Minimum objectness × class-prob score for inference.
        nms_thresh:   IoU threshold for NMS at inference.
        lambda_coord: Up-weighting for box regression loss.
        lambda_noobj: Down-weighting for no-object confidence loss.
        tiny:         Use the tiny Darknet backbone.
    """

    model_type: ClassVar[str] = "yolo_v2"

    num_classes: int = 80
    in_channels: int = 3
    num_anchors: int = 5
    anchors: tuple[tuple[float, float], ...] = field(default=_COCO_ANCHORS)
    score_thresh: float = 0.5
    nms_thresh: float = 0.5
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    tiny: bool = False

    def __post_init__(self) -> None:
        if len(self.anchors) != self.num_anchors:
            raise ValueError(
                f"anchors length ({len(self.anchors)}) must equal "
                f"num_anchors ({self.num_anchors})"
            )
        object.__setattr__(
            self, "anchors", tuple((float(a), float(b)) for a, b in self.anchors)
        )


# ---------------------------------------------------------------------------
# Darknet-19 backbone building blocks
# ---------------------------------------------------------------------------


def _conv_bn_lrelu(
    in_ch: int, out_ch: int, kernel: int, stride: int = 1, padding: int = 0
) -> nn.Sequential:
    """Conv2d → BatchNorm2d → LeakyReLU(0.1)."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(negative_slope=0.1),
    )


class _Darknet19(nn.Module):
    """Darknet-19 backbone.

    Returns two feature maps:
      - ``route``: (B, 512, H/16, W/16)  — for passthrough at stride 16
      - ``out``:   (B, 1024, H/32, W/32) — main features at stride 32
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Stage 1: 3→32, pool
        self.stage1 = nn.Sequential(
            _conv_bn_lrelu(in_channels, 32, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 2: 32→64, pool
        self.stage2 = nn.Sequential(
            _conv_bn_lrelu(32, 64, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 3: 64→128→64→128, pool
        self.stage3 = nn.Sequential(
            _conv_bn_lrelu(64, 128, 3, padding=1),
            _conv_bn_lrelu(128, 64, 1),
            _conv_bn_lrelu(64, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 4: 128→256→128→256, pool
        self.stage4 = nn.Sequential(
            _conv_bn_lrelu(128, 256, 3, padding=1),
            _conv_bn_lrelu(256, 128, 1),
            _conv_bn_lrelu(128, 256, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 5: 256→512→256→512→256→512 (passthrough hook at stride-16)
        self.stage5 = nn.Sequential(
            _conv_bn_lrelu(256, 512, 3, padding=1),
            _conv_bn_lrelu(512, 256, 1),
            _conv_bn_lrelu(256, 512, 3, padding=1),
            _conv_bn_lrelu(512, 256, 1),
            _conv_bn_lrelu(256, 512, 3, padding=1),
        )
        # Pool between stage5 and stage6
        self.pool56 = nn.MaxPool2d(2, stride=2)
        # Stage 6: 512→1024→512→1024→512→1024 (output at stride-32)
        self.stage6 = nn.Sequential(
            _conv_bn_lrelu(512, 1024, 3, padding=1),
            _conv_bn_lrelu(1024, 512, 1),
            _conv_bn_lrelu(512, 1024, 3, padding=1),
            _conv_bn_lrelu(1024, 512, 1),
            _conv_bn_lrelu(512, 1024, 3, padding=1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        """Run Darknet-19 and return (route, out).

        Returns:
            (route, out):
                route: (B, 512, H/16, W/16)
                out:   (B, 1024, H/32, W/32)
        """
        x = cast(Tensor, self.stage1(x))
        x = cast(Tensor, self.stage2(x))
        x = cast(Tensor, self.stage3(x))
        x = cast(Tensor, self.stage4(x))
        route = cast(Tensor, self.stage5(x))  # (B, 512, H/16, W/16)
        x = cast(Tensor, self.pool56(route))
        out = cast(Tensor, self.stage6(x))  # (B, 1024, H/32, W/32)
        return route, out


class _Darknet19Tiny(nn.Module):
    """Lightweight Darknet-19 backbone without bottleneck layers in stages 3–5.

    The structure is designed so that:
      - ``route`` is captured at stride-16 (H/16 × W/16) with 512 channels,
        matching the passthrough expectation of the full Darknet-19.
      - ``out`` is at stride-32 (H/32 × W/32) with 1024 channels.

    Returns:
        (route, out):
            route: (B, 512, H/16, W/16)
            out:   (B, 1024, H/32, W/32)
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Four pooling stages → stride-16 (H/16, W/16) entering stage5
        self.stage1 = nn.Sequential(
            _conv_bn_lrelu(in_channels, 16, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.stage2 = nn.Sequential(
            _conv_bn_lrelu(16, 32, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.stage3 = nn.Sequential(
            _conv_bn_lrelu(32, 64, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        self.stage4 = nn.Sequential(
            _conv_bn_lrelu(64, 128, 3, padding=1),
            nn.MaxPool2d(2, stride=2),
        )
        # Stage 5 at stride-16: 128→256→512 (passthrough route hook)
        self.stage5 = nn.Sequential(
            _conv_bn_lrelu(128, 256, 3, padding=1),
            _conv_bn_lrelu(256, 512, 3, padding=1),
        )
        # Pool56: stride-16 → stride-32
        self.pool56 = nn.MaxPool2d(2, stride=2)
        # Stage 6 at stride-32: 512→1024 (main output)
        self.stage6 = nn.Sequential(
            _conv_bn_lrelu(512, 1024, 3, padding=1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        x = cast(Tensor, self.stage1(x))  # stride-2
        x = cast(Tensor, self.stage2(x))  # stride-4
        x = cast(Tensor, self.stage3(x))  # stride-8
        x = cast(Tensor, self.stage4(x))  # stride-16
        route = cast(Tensor, self.stage5(x))  # stride-16, 512ch  ← passthrough
        x = cast(Tensor, self.pool56(route))  # stride-32
        out = cast(Tensor, self.stage6(x))  # stride-32, 1024ch
        return route, out


# ---------------------------------------------------------------------------
# Space-to-depth (passthrough layer)
# ---------------------------------------------------------------------------


def _space_to_depth(x: Tensor, block_size: int) -> Tensor:
    """Rearrange spatial blocks into the channel dimension (passthrough).

    Equivalent to ``nn.PixelUnshuffle`` or ``tf.nn.space_to_depth``.

    Args:
        x:          (B, C, H, W) — H and W must be divisible by block_size.
        block_size: Downscaling factor (typically 2).

    Returns:
        (B, C*block_size², H//block_size, W//block_size)
    """
    B = int(x.shape[0])
    C = int(x.shape[1])
    H = int(x.shape[2])
    W = int(x.shape[3])
    bs = block_size
    assert H % bs == 0 and W % bs == 0, "H and W must be divisible by block_size"

    new_h = H // bs
    new_w = W // bs
    new_c = C * bs * bs

    # (B, C, H, W)
    # → (B, C, new_h, bs, new_w, bs)
    x = x.reshape(B, C, new_h, bs, new_w, bs)
    # → (B, new_h, new_w, C, bs, bs)
    x = x.permute(0, 2, 4, 1, 3, 5)
    # → (B, new_h, new_w, new_c)
    x = x.reshape(B, new_h, new_w, new_c)
    # → (B, new_c, new_h, new_w)
    return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# YOLOv2 model
# ---------------------------------------------------------------------------


class YOLOV2ForObjectDetection(PretrainedModel):
    """YOLOv2 / YOLO9000 object detector (Redmon & Farhadi, CVPR 2017).

    Args:
        config: :class:`YOLOV2Config` controlling architecture and hyperparams.
    """

    config_class: ClassVar[type[YOLOV2Config]] = YOLOV2Config
    base_model_prefix: ClassVar[str] = "backbone"

    config: YOLOV2Config

    def __init__(self, config: YOLOV2Config) -> None:
        super().__init__(config)
        A = config.num_anchors
        C = config.num_classes

        # Backbone
        if config.tiny:
            self.backbone: _Darknet19 | _Darknet19Tiny = _Darknet19Tiny(
                config.in_channels
            )
            route_ch = 512
        else:
            self.backbone = _Darknet19(config.in_channels)
            route_ch = 512

        # Detection head conv layers (applied to stride-32 feature map)
        self.det1 = _conv_bn_lrelu(1024, 1024, 3, padding=1)
        self.det2 = _conv_bn_lrelu(1024, 1024, 3, padding=1)

        # Passthrough: route_ch * 4 (space-to-depth s=2) + 1024
        pt_ch = route_ch * 4  # 512 * 4 = 2048
        merged_ch = pt_ch + 1024  # 3072

        self.det3 = _conv_bn_lrelu(merged_ch, 1024, 1)
        # Final prediction conv: no BN, no activation
        self.pred = nn.Conv2d(1024, A * (5 + C), 1)

    def _forward_raw(self, x: Tensor) -> Tensor:
        """Run backbone + detection head → raw prediction tensor.

        Returns:
            (B, A*(5+C), H/32, W/32)
        """
        route, feat = self.backbone.forward(x)  # (B,512,H/16,W/16), (B,1024,H/32,W/32)
        feat = cast(Tensor, self.det1(feat))  # (B, 1024, H/32, W/32)
        feat = cast(Tensor, self.det2(feat))

        # Passthrough: space_to_depth on route → (B, 2048, H/32, W/32)
        passthrough = _space_to_depth(route, 2)

        # Concatenate along channel axis
        feat = lucid.cat([passthrough, feat], dim=1)  # (B, 3072, H/32, W/32)
        feat = cast(Tensor, self.det3(feat))  # (B, 1024, H/32, W/32)
        return cast(Tensor, self.pred(feat))  # (B, A*(5+C), H/32, W/32)

    def _decode_predictions(
        self,
        raw: Tensor,
        image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Decode raw anchor predictions to flat predictions.

        Args:
            raw:        (B, A*(5+C), fH, fW) from the prediction conv.
            image_size: (H, W) of the input image in pixels.

        Returns:
            (logits, pred_boxes):
                logits:     (B, fH*fW*A, C) raw class scores.
                pred_boxes: (B, fH*fW*A, 4) decoded xyxy pixel boxes.
        """
        cfg = self.config
        A = cfg.num_anchors
        C = cfg.num_classes
        anchors = cfg.anchors
        H, W = image_size

        B_batch = int(raw.shape[0])
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])

        # Stride of this feature map
        stride_h = H / fH
        stride_w = W / fW

        # (B, A*(5+C), fH, fW) → (B, A, 5+C, fH, fW) → (B, fH, fW, A, 5+C)
        raw = raw.reshape(B_batch, A, 5 + C, fH, fW)
        raw = raw.permute(0, 3, 4, 1, 2)  # (B, fH, fW, A, 5+C)

        # tx, ty → sigmoid; tw, th → raw; conf → sigmoid; cls → raw
        tx = F.sigmoid(raw[..., 0])  # (B, fH, fW, A)
        ty = F.sigmoid(raw[..., 1])  # (B, fH, fW, A)
        tw = raw[..., 2]  # (B, fH, fW, A)
        th = raw[..., 3]  # (B, fH, fW, A)
        raw_cls = raw[..., 5:]  # (B, fH, fW, A, C)

        # Build cell offset grids
        # col_offsets: (1, 1, fW, 1) — broadcast over B, fH, A
        col_data: list[float] = [float(c) for c in range(fW)]
        row_data: list[float] = [float(r) for r in range(fH)]

        col_offsets = lucid.tensor(col_data).reshape(1, 1, fW, 1)  # (1,1,fW,1)
        row_offsets = lucid.tensor(row_data).reshape(1, fH, 1, 1)  # (1,fH,1,1)

        # Decode box centres in pixels
        # bx = (sigmoid(tx) + col) * stride_w
        # by = (sigmoid(ty) + row) * stride_h
        bx = (tx + col_offsets) * stride_w  # (B, fH, fW, A)
        by = (ty + row_offsets) * stride_h  # (B, fH, fW, A)

        # Decode box sizes in pixels
        # bw = anchor_w * exp(tw) * stride_w
        # bh = anchor_h * exp(th) * stride_h
        anchor_w_data = [anchors[a][0] for a in range(A)]
        anchor_h_data = [anchors[a][1] for a in range(A)]

        aw_t = lucid.tensor(anchor_w_data).reshape(1, 1, 1, A)  # (1,1,1,A)
        ah_t = lucid.tensor(anchor_h_data).reshape(1, 1, 1, A)

        bw = aw_t * lucid.exp(tw.clamp(min=-10.0, max=10.0)) * stride_w
        bh = ah_t * lucid.exp(th.clamp(min=-10.0, max=10.0)) * stride_h

        # Convert cxcywh → xyxy
        x1 = bx - bw / 2.0
        y1 = by - bh / 2.0
        x2 = bx + bw / 2.0
        y2 = by + bh / 2.0

        # Clamp to image
        x1 = x1.clamp(min=0.0, max=float(W))
        y1 = y1.clamp(min=0.0, max=float(H))
        x2 = x2.clamp(min=0.0, max=float(W))
        y2 = y2.clamp(min=0.0, max=float(H))

        # Stack boxes: (B, fH, fW, A, 4)
        boxes = lucid.stack([x1, y1, x2, y2], dim=-1)

        # Flatten spatial and anchor dims: (B, fH*fW*A, 4)
        pred_boxes = boxes.reshape(B_batch, fH * fW * A, 4)

        # Class logits: (B, fH, fW, A, C) → (B, fH*fW*A, C)
        logits = raw_cls.reshape(B_batch, fH * fW * A, C)

        return logits, pred_boxes

    def _compute_loss(
        self,
        raw: Tensor,
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> Tensor:
        """Compute YOLOv2 multi-part MSE/CE loss.

        Args:
            raw:        (B, A*(5+C), fH, fW) raw network output.
            targets:    List of B dicts with "boxes" (M,4) xyxy [0,1]
                        and "labels" (M,) int.
            image_size: (H, W) in pixels.

        Returns:
            Scalar loss tensor.
        """
        cfg = self.config
        A = cfg.num_anchors
        C = cfg.num_classes
        anchors = cfg.anchors
        lc = cfg.lambda_coord
        ln = cfg.lambda_noobj
        H, W = image_size

        B_batch = int(raw.shape[0])
        fH = int(raw.shape[2])
        fW = int(raw.shape[3])

        stride_h = H / fH
        stride_w = W / fW

        # Reshape raw: (B, fH, fW, A, 5+C)
        raw_r = raw.reshape(B_batch, A, 5 + C, fH, fW)
        raw_r = raw_r.permute(0, 3, 4, 1, 2)

        loss_parts: list[Tensor] = []

        for bi in range(B_batch):
            gt_boxes_norm = targets[bi]["boxes"]  # (M, 4) xyxy in [0,1]
            gt_labels = targets[bi]["labels"]  # (M,)
            M = int(gt_boxes_norm.shape[0])

            # Convert GT boxes to pixel space
            gt_x1 = gt_boxes_norm[:, 0] * float(W)
            gt_y1 = gt_boxes_norm[:, 1] * float(H)
            gt_x2 = gt_boxes_norm[:, 2] * float(W)
            gt_y2 = gt_boxes_norm[:, 3] * float(H)
            gt_cx = (gt_x1 + gt_x2) / 2.0
            gt_cy = (gt_y1 + gt_y2) / 2.0
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1

            # Assign each GT to best-matching anchor (by wh IoU, ignoring position)
            # Responsible cell: floor(cx/stride_w), floor(cy/stride_h)
            # responsible anchor: highest IoU between GT wh and anchor wh (both centred)

            assigned: dict[
                tuple[int, int, int], tuple[float, float, float, float, int]
            ] = {}

            for m in range(M):
                cx_m = float(gt_cx[m].item())
                cy_m = float(gt_cy[m].item())
                w_m = float(gt_w[m].item())
                h_m = float(gt_h[m].item())
                cls_m = int(gt_labels[m].item())

                col = min(int(cx_m / stride_w), fW - 1)
                row = min(int(cy_m / stride_h), fH - 1)

                # Find best anchor by wh IoU (anchors in pixel units via stride)
                best_a = 0
                best_iou = -1.0
                for a in range(A):
                    aw = anchors[a][0] * stride_w
                    ah = anchors[a][1] * stride_h

                    inter_w = min(w_m, aw)
                    inter_h = min(h_m, ah)
                    inter = inter_w * inter_h
                    union = w_m * h_m + aw * ah - inter
                    wh_iou = inter / max(union, 1e-6)
                    if wh_iou > best_iou:
                        best_iou = wh_iou
                        best_a = a

                assigned[(row, col, best_a)] = (cx_m, cy_m, w_m, h_m, cls_m)

            xy_terms: list[Tensor] = []
            wh_terms: list[Tensor] = []
            conf_obj: list[Tensor] = []
            conf_noobj: list[Tensor] = []
            cls_terms: list[Tensor] = []

            for row in range(fH):
                for col in range(fW):
                    for a in range(A):
                        raw_cell = raw_r[bi, row, col, a, :]  # (5+C,)

                        raw_tx = raw_cell[0]
                        raw_ty = raw_cell[1]
                        raw_tw = raw_cell[2]
                        raw_th = raw_cell[3]
                        raw_conf = raw_cell[4]
                        raw_cls = raw_cell[5:]  # (C,)

                        if (row, col, a) in assigned:
                            cx_m, cy_m, w_m, h_m, cls_m = assigned[(row, col, a)]

                            # tx, ty targets: inverse sigmoid of fractional cell offset
                            tgt_tx_rel = cx_m / stride_w - float(col)
                            tgt_ty_rel = cy_m / stride_h - float(row)
                            tgt_tx_rel = max(1e-6, min(1.0 - 1e-6, tgt_tx_rel))
                            tgt_ty_rel = max(1e-6, min(1.0 - 1e-6, tgt_ty_rel))

                            sig_tx = F.sigmoid(raw_tx)
                            sig_ty = F.sigmoid(raw_ty)
                            xy_terms.append(
                                (sig_tx - lucid.tensor([tgt_tx_rel])[0]) ** 2
                            )
                            xy_terms.append(
                                (sig_ty - lucid.tensor([tgt_ty_rel])[0]) ** 2
                            )

                            # tw, th targets: log(gt / anchor)
                            aw = anchors[a][0] * stride_w
                            ah = anchors[a][1] * stride_h
                            tgt_tw = math.log(max(w_m, 1e-6) / max(aw, 1e-6))
                            tgt_th = math.log(max(h_m, 1e-6) / max(ah, 1e-6))
                            wh_terms.append((raw_tw - lucid.tensor([tgt_tw])[0]) ** 2)
                            wh_terms.append((raw_th - lucid.tensor([tgt_th])[0]) ** 2)

                            # Confidence target = 1.0
                            sig_conf = F.sigmoid(raw_conf)
                            conf_obj.append((sig_conf - lucid.ones((1,))[0]) ** 2)

                            # Class loss: MSE vs one-hot
                            tgt_cls_list = [0.0] * C
                            tgt_cls_list[cls_m] = 1.0
                            tgt_cls_t = lucid.tensor(tgt_cls_list)
                            cls_terms.append(((raw_cls - tgt_cls_t) ** 2).sum())

                        else:
                            # No-object: confidence should be 0
                            sig_conf = F.sigmoid(raw_conf)
                            conf_noobj.append(sig_conf**2)

            def _mean_or_zero(parts: list[Tensor]) -> Tensor:
                if not parts:
                    return lucid.zeros((1,))
                return lucid.cat([t.reshape(1) for t in parts]).mean()

            loss_xy = lc * _mean_or_zero(xy_terms)
            loss_wh = lc * _mean_or_zero(wh_terms)
            loss_conf_o = _mean_or_zero(conf_obj)
            loss_conf_n = ln * _mean_or_zero(conf_noobj)
            loss_cls = _mean_or_zero(cls_terms)

            img_loss: Tensor = loss_xy + loss_wh + loss_conf_o + loss_conf_n + loss_cls
            loss_parts.append(img_loss.reshape(1))

        return lucid.cat(loss_parts).mean()

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: list[dict[str, Tensor]] | None = None,
        image_size: tuple[int, int] = (416, 416),
    ) -> ObjectDetectionOutput:
        """Run YOLOv2 forward pass.

        Args:
            x:          (B, C, H, W) input image tensor.
            targets:    Optional list of B dicts for training loss.
                        Each dict has:
                        - ``"boxes"``:  (M_i, 4) xyxy boxes normalised to [0,1].
                        - ``"labels"``: (M_i,) integer class indices.
            image_size: (H, W) of the input image in pixels (default 416×416).

        Returns:
            :class:`~lucid.models._output.ObjectDetectionOutput` with:
            - ``logits``:     (B, fH*fW*A, num_classes) raw class scores
            - ``pred_boxes``: (B, fH*fW*A, 4) decoded xyxy pixel boxes
            - ``loss``:       scalar tensor when targets are provided, else None
        """
        raw = self._forward_raw(x)
        logits, pred_boxes = self._decode_predictions(raw, image_size)

        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(raw, targets, image_size)

        return ObjectDetectionOutput(logits=logits, pred_boxes=pred_boxes, loss=loss)

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_size: tuple[int, int] = (416, 416),
    ) -> list[dict[str, Tensor]]:
        """Per-class NMS on raw predictions.

        Args:
            output:     Output from :meth:`forward`.
            image_size: (H, W) of the input image.

        Returns:
            List of per-image result dicts with ``"boxes"``, ``"scores"``,
            ``"labels"``.
        """
        cfg = self.config
        B_batch = int(output.logits.shape[0])
        C = cfg.num_classes
        results: list[dict[str, Tensor]] = []

        for b in range(B_batch):
            lg_b = output.logits[b]  # (fH*fW*A, C)
            bx_b = output.pred_boxes[b]  # (fH*fW*A, 4)

            # Class probabilities via softmax
            sc_b = F.softmax(lg_b, dim=-1)  # (N, C)

            keep_boxes: list[Tensor] = []
            keep_scores: list[Tensor] = []
            keep_labels: list[Tensor] = []

            for c in range(C):
                sc_c = sc_b[:, c]
                mask: list[int] = [
                    i
                    for i in range(int(sc_c.shape[0]))
                    if float(sc_c[i].item()) >= cfg.score_thresh
                ]
                if not mask:
                    continue
                mask_t = lucid.tensor(mask)
                sc_sel = sc_c[mask_t]
                bx_sel = bx_b[mask_t]
                keep = batched_nms(
                    bx_sel,
                    sc_sel,
                    lucid.zeros(int(sc_sel.shape[0])),
                    cfg.nms_thresh,
                )
                keep_boxes.append(bx_sel[keep])
                keep_scores.append(sc_sel[keep])
                keep_labels.append(lucid.full((int(keep.shape[0]),), float(c)))

            if keep_boxes:
                results.append(
                    {
                        "boxes": lucid.cat(keep_boxes, dim=0),
                        "scores": lucid.cat(keep_scores, dim=0),
                        "labels": lucid.cat(keep_labels, dim=0),
                    }
                )
            else:
                results.append(
                    {
                        "boxes": lucid.zeros((0, 4)),
                        "scores": lucid.zeros((0,)),
                        "labels": lucid.zeros((0,)),
                    }
                )

        return results


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

_CFG_V2 = YOLOV2Config(
    num_classes=80,
    in_channels=3,
    num_anchors=5,
    anchors=_COCO_ANCHORS,
    score_thresh=0.5,
    nms_thresh=0.5,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    tiny=False,
)

_CFG_V2_TINY = YOLOV2Config(
    num_classes=80,
    in_channels=3,
    num_anchors=5,
    anchors=_COCO_ANCHORS,
    score_thresh=0.5,
    nms_thresh=0.5,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    tiny=True,
)


def _make_v2(
    cfg: YOLOV2Config, overrides: dict[str, object]
) -> YOLOV2ForObjectDetection:
    if overrides:
        merged = {**cfg.__dict__, **overrides}
        merged.pop("model_type", None)
        cfg = YOLOV2Config(**merged)
    return YOLOV2ForObjectDetection(cfg)


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v2",
    model_class=YOLOV2ForObjectDetection,
    default_config=_CFG_V2,
)
def yolo_v2(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOV2ForObjectDetection:
    """YOLOv2 (YOLO9000) with Darknet-19 backbone (Redmon & Farhadi, CVPR 2017).

    Anchor-based single-shot detector with passthrough layer and 5 COCO
    anchor clusters.  Runs at 13×13 (stride 32) output resolution by default.
    """
    return _make_v2(_CFG_V2, overrides)


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v2",
    model_class=YOLOV2ForObjectDetection,
    default_config=_CFG_V2_TINY,
)
def yolo_v2_tiny(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOV2ForObjectDetection:
    """YOLOv2 with tiny Darknet backbone — lightweight variant.

    Uses a compact Darknet backbone without bottleneck layers for faster
    inference at the cost of detection accuracy.
    """
    return _make_v2(_CFG_V2_TINY, overrides)
