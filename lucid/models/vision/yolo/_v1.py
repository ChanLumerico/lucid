"""YOLOv1 — You Only Look Once (Redmon et al., CVPR 2016).

Paper: "You Only Look Once: Unified, Real-Time Object Detection"

Key ideas
----------
- Divide the image into an S×S grid; each cell predicts B bounding boxes
  and C class probabilities simultaneously in a single forward pass.
- Each box prediction consists of (cx, cy, w, h, conf) where cx/cy are
  offsets relative to the cell, and w/h are relative to the image size.
- At inference, class-specific confidence scores filter predictions via NMS.
- Training loss is a multi-part MSE loss with λ_coord=5 (up-weights box
  regression) and λ_noobj=0.5 (down-weights no-object confidence).

Architecture (Darknet backbone from the paper)
----------------------------------------------
  Image → Darknet conv layers → AdaptiveAvgPool2d(S,S) → flatten
        → FC(1024·S², 4096) → LeakyReLU(0.1)
        → FC(4096, S²·(B·5+C))

Each conv block uses Conv2d → BatchNorm2d → LeakyReLU(0.1).
"M" entries become MaxPool2d(2,2); list entries are repeated conv groups.

Losses (training)
-----------------
Requires ``targets`` — list of B dicts with:
  ``"boxes"``  : (M_i, 4) xyxy, normalised to [0, 1] by image H/W.
  ``"labels"`` : (M_i,)  integer foreground class ids (0-indexed).

Multi-part MSE loss matching the paper §2.2:
  L = λ_coord · L_xy  +  λ_coord · L_wh  +  L_conf_obj
    +  λ_noobj · L_conf_noobj  +  L_cls
"""

import math
from dataclasses import dataclass
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

# Darknet backbone architecture spec used by the full model
_ARCH_FULL: list[object] = [
    (64, 7, 2, 3),
    "M",
    (192, 3, 1, 1),
    "M",
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    "M",
    [(256, 1, 1, 0), (512, 3, 1, 1), 4],
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    "M",
    [(512, 1, 1, 0), (1024, 3, 1, 1), 2],
    (1024, 3, 1, 1),
    (1024, 3, 2, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]

# Tiny Darknet architecture spec
_ARCH_TINY: list[object] = [
    (16, 3, 1, 1),
    "M",
    (32, 3, 1, 1),
    "M",
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    "M",
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
    (1024, 3, 1, 1),
]


@dataclass(frozen=True)
class YOLOV1Config(ModelConfig):
    """Configuration for YOLOv1 (Redmon et al., CVPR 2016).

    Args:
        num_classes:  Number of object classes C (COCO default: 80).
        in_channels:  Input image channels.
        split_size:   Grid cells per side S (default 7 → 7×7 grid).
        num_boxes:    Bounding boxes predicted per cell B (default 2).
        lambda_coord: Up-weighting factor for box coordinate loss.
        lambda_noobj: Down-weighting factor for no-object confidence loss.
        score_thresh: Minimum class-confidence score at inference.
        nms_thresh:   IoU threshold for NMS.
        tiny:         Use the tiny Darknet backbone instead of the full one.
    """

    model_type: ClassVar[str] = "yolo_v1"

    num_classes: int = 80
    in_channels: int = 3
    split_size: int = 7
    num_boxes: int = 2
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5
    score_thresh: float = 0.5
    nms_thresh: float = 0.5
    tiny: bool = False


# ---------------------------------------------------------------------------
# Darknet backbone
# ---------------------------------------------------------------------------


def _conv_block(
    in_ch: int, out_ch: int, kernel: int, stride: int, padding: int
) -> nn.Sequential:
    """Conv2d → BatchNorm2d → LeakyReLU(0.1) block."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.LeakyReLU(negative_slope=0.1),
    )


def _build_darknet(
    in_channels: int,
    arch: list[object],
) -> tuple[nn.Sequential, int]:
    """Build the Darknet convolutional backbone from an architecture spec.

    Returns:
        (sequential, out_channels): The backbone module and its output channels.
    """
    layers: list[nn.Module] = []
    ch = in_channels

    for entry in arch:
        if entry == "M":
            layers.append(nn.MaxPool2d(2, stride=2))
        elif isinstance(entry, tuple):
            out_ch, k, s, p = entry
            layers.append(_conv_block(ch, out_ch, k, s, p))
            ch = out_ch
        elif isinstance(entry, list):
            # [conv1_spec, conv2_spec, repeats]
            # The list ends with an int (repeat count); preceding elements are specs
            repeats = int(entry[-1])
            specs: list[tuple[int, int, int, int]] = [
                cast(tuple[int, int, int, int], e) for e in entry[:-1]
            ]
            for _ in range(repeats):
                for spec in specs:
                    out_ch, k, s, p = spec
                    layers.append(_conv_block(ch, out_ch, k, s, p))
                    ch = out_ch
        else:
            raise ValueError(f"Unknown arch entry: {entry!r}")

    return nn.Sequential(*layers), ch


# ---------------------------------------------------------------------------
# YOLOv1 model
# ---------------------------------------------------------------------------


class YOLOV1ForObjectDetection(PretrainedModel):
    """YOLOv1 object detector (Redmon et al., CVPR 2016).

    Args:
        config: :class:`YOLOV1Config` controlling architecture and loss hyperparams.
    """

    config_class: ClassVar[type[YOLOV1Config]] = YOLOV1Config
    base_model_prefix: ClassVar[str] = "darknet"

    config: YOLOV1Config

    def __init__(self, config: YOLOV1Config) -> None:
        super().__init__(config)
        S = config.split_size
        B = config.num_boxes
        C = config.num_classes

        arch: list[object] = _ARCH_TINY if config.tiny else _ARCH_FULL
        self.darknet, backbone_ch = _build_darknet(config.in_channels, arch)

        self.pool = nn.AdaptiveAvgPool2d((S, S))
        flat = backbone_ch * S * S
        self.fc1 = nn.Linear(flat, 4096)
        self.act1 = nn.LeakyReLU(negative_slope=0.1)
        self.drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4096, S * S * (B * 5 + C))

    def _forward_features(self, x: Tensor) -> Tensor:
        """Run backbone + pool + FC → raw grid predictions.

        Returns:
            (B_batch, S, S, num_boxes*5 + num_classes) raw predictions.
        """
        cfg = self.config
        S = cfg.split_size
        B_boxes = cfg.num_boxes
        C = cfg.num_classes
        batch = int(x.shape[0])

        feat = cast(Tensor, self.darknet(x))  # (N, ch, H', W')
        feat = cast(Tensor, self.pool(feat))  # (N, ch, S, S)
        feat = feat.flatten(1)  # (N, ch*S*S)
        feat = cast(Tensor, self.act1(cast(Tensor, self.fc1(feat))))
        feat = cast(Tensor, self.drop(feat))
        feat = cast(Tensor, self.fc2(feat))  # (N, S*S*(B*5+C))
        return feat.reshape(batch, S, S, B_boxes * 5 + C)

    def _decode_predictions(
        self,
        raw: Tensor,
        image_size: tuple[int, int],
    ) -> tuple[Tensor, Tensor]:
        """Decode raw grid output to flat predictions.

        Args:
            raw:        (B_batch, S, S, num_boxes*5+C) raw network output.
            image_size: (H, W) of the input image in pixels.

        Returns:
            (logits, pred_boxes):
                logits:     (B_batch, S*S*num_boxes, C) raw class scores.
                pred_boxes: (B_batch, S*S*num_boxes, 4) decoded xyxy pixel boxes.
        """
        cfg = self.config
        S = cfg.split_size
        B_boxes = cfg.num_boxes
        H, W = image_size
        batch = int(raw.shape[0])
        cell_h = H / S
        cell_w = W / S

        # raw: (batch, S, S, B*5+C)
        # split into per-box predictions and class scores
        # layout: [box0_cx, box0_cy, box0_w, box0_h, box0_conf, box1_..., ..., cls0, cls1, ...]
        box_preds = raw[..., : B_boxes * 5]  # (batch, S, S, B*5)
        cls_preds = raw[..., B_boxes * 5 :]  # (batch, S, S, C)

        # Reshape box preds: (batch, S, S, B, 5)
        box_preds = box_preds.reshape(batch, S, S, B_boxes, 5)

        all_boxes_list: list[Tensor] = []
        all_logits_list: list[Tensor] = []

        for r in range(S):
            for c in range(S):
                # box_cell: (batch, B, 5)
                box_cell = box_preds[:, r, c, :, :]
                # cls_cell: (batch, C)
                cls_cell = cls_preds[:, r, c, :]

                # Decode each box in this cell
                for b in range(B_boxes):
                    raw_cx = box_cell[:, b, 0]  # (batch,)
                    raw_cy = box_cell[:, b, 1]
                    raw_w = box_cell[:, b, 2]
                    raw_h = box_cell[:, b, 3]

                    # cx/cy: sigmoid + cell offset → pixel x_center
                    x_pixel = (F.sigmoid(raw_cx) + float(c)) * cell_w
                    y_pixel = (F.sigmoid(raw_cy) + float(r)) * cell_h
                    # w/h: exp → pixel dimensions (clamp to prevent overflow)
                    w_pixel = lucid.exp(raw_w.clamp(min=-10.0, max=10.0)) * float(W)
                    h_pixel = lucid.exp(raw_h.clamp(min=-10.0, max=10.0)) * float(H)

                    # Convert cxcywh → xyxy
                    x1 = x_pixel - w_pixel / 2.0
                    y1 = y_pixel - h_pixel / 2.0
                    x2 = x_pixel + w_pixel / 2.0
                    y2 = y_pixel + h_pixel / 2.0

                    # Clamp to image boundaries
                    x1 = x1.clamp(min=0.0, max=float(W))
                    y1 = y1.clamp(min=0.0, max=float(H))
                    x2 = x2.clamp(min=0.0, max=float(W))
                    y2 = y2.clamp(min=0.0, max=float(H))

                    # box: (batch, 4)
                    box = lucid.stack([x1, y1, x2, y2], dim=1)
                    all_boxes_list.append(box.unsqueeze(1))  # (batch, 1, 4)
                    all_logits_list.append(cls_cell.unsqueeze(1))  # (batch, 1, C)

        # Stack: (batch, S*S*B, 4) and (batch, S*S*B, C)
        pred_boxes = lucid.cat(all_boxes_list, dim=1)  # (batch, S*S*B, 4)
        logits = lucid.cat(all_logits_list, dim=1)  # (batch, S*S*B, C)

        return logits, pred_boxes

    def _compute_loss(
        self,
        raw: Tensor,
        targets: list[dict[str, Tensor]],
        image_size: tuple[int, int],
    ) -> Tensor:
        """Compute the YOLOv1 multi-part MSE loss.

        Args:
            raw:        (B_batch, S, S, B*5+C) raw network output.
            targets:    List of B dicts with "boxes" (M,4) xyxy in [0,1]
                        and "labels" (M,) int.
            image_size: (H, W) in pixels.

        Returns:
            Scalar loss tensor.
        """
        cfg = self.config
        S = cfg.split_size
        B_boxes = cfg.num_boxes
        C = cfg.num_classes
        lc = cfg.lambda_coord
        ln = cfg.lambda_noobj
        H, W = image_size
        batch_size = int(raw.shape[0])

        # raw layout: (batch, S, S, B*5 + C)
        box_preds = raw[..., : B_boxes * 5].reshape(batch_size, S, S, B_boxes, 5)
        cls_preds = raw[..., B_boxes * 5 :]  # (batch, S, S, C)

        loss_parts: list[Tensor] = []

        for bi in range(batch_size):
            gt_boxes_norm = targets[bi]["boxes"]  # (M, 4) xyxy in [0,1]
            gt_labels = targets[bi]["labels"]  # (M,)
            M = int(gt_boxes_norm.shape[0])

            # Convert GT boxes from [0,1] xyxy → pixel cxcywh
            gt_x1 = gt_boxes_norm[:, 0] * float(W)
            gt_y1 = gt_boxes_norm[:, 1] * float(H)
            gt_x2 = gt_boxes_norm[:, 2] * float(W)
            gt_y2 = gt_boxes_norm[:, 3] * float(H)
            gt_cx = (gt_x1 + gt_x2) / 2.0
            gt_cy = (gt_y1 + gt_y2) / 2.0
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1

            # For each GT object, determine responsible cell and box
            # cell assignment: floor(cx / cell_w), floor(cy / cell_h)
            cell_w_px = W / S
            cell_h_px = H / S

            # Track which (r, c, b) slots are assigned
            # Value: (cx_pixel, cy_pixel, w_pixel, h_pixel, iou, class_idx)
            assigned: dict[
                tuple[int, int, int], tuple[float, float, float, float, float, int]
            ] = {}

            for m in range(M):
                cx_m = float(gt_cx[m].item())
                cy_m = float(gt_cy[m].item())
                w_m = float(gt_w[m].item())
                h_m = float(gt_h[m].item())
                cls_m = int(gt_labels[m].item())

                col = min(int(cx_m / cell_w_px), S - 1)
                row = min(int(cy_m / cell_h_px), S - 1)

                # Convert GT to xyxy pixel for IoU computation
                gx1 = cx_m - w_m / 2.0
                gy1 = cy_m - h_m / 2.0
                gx2 = cx_m + w_m / 2.0
                gy2 = cy_m + h_m / 2.0

                # Find the predicted box with highest IoU in this cell
                best_b = 0
                best_iou = -1.0
                for b in range(B_boxes):
                    raw_cx_b = float(box_preds[bi, row, col, b, 0].item())
                    raw_cy_b = float(box_preds[bi, row, col, b, 1].item())
                    raw_w_b = float(box_preds[bi, row, col, b, 2].item())
                    raw_h_b = float(box_preds[bi, row, col, b, 3].item())

                    px = (1.0 / (1.0 + math.exp(-raw_cx_b)) + col) * cell_w_px
                    py = (1.0 / (1.0 + math.exp(-raw_cy_b)) + row) * cell_h_px
                    pw = math.exp(max(-10.0, min(10.0, raw_w_b))) * W
                    ph = math.exp(max(-10.0, min(10.0, raw_h_b))) * H

                    px1 = px - pw / 2.0
                    py1 = py - ph / 2.0
                    px2 = px + pw / 2.0
                    py2 = py + ph / 2.0

                    # Simple scalar IoU
                    ix1 = max(px1, gx1)
                    iy1 = max(py1, gy1)
                    ix2 = min(px2, gx2)
                    iy2 = min(py2, gy2)
                    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                    union = (px2 - px1) * (py2 - py1) + w_m * h_m - inter
                    iou_val = inter / max(union, 1e-6)

                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_b = b

                assigned[(row, col, best_b)] = (
                    cx_m,
                    cy_m,
                    w_m,
                    h_m,
                    float(best_iou),
                    cls_m,
                )

            # Now compute loss contributions
            xy_terms: list[Tensor] = []
            wh_terms: list[Tensor] = []
            conf_obj: list[Tensor] = []
            conf_noobj: list[Tensor] = []
            cls_terms: list[Tensor] = []

            for row in range(S):
                for col in range(S):
                    for b in range(B_boxes):
                        pred_cx = box_preds[bi, row, col, b, 0]
                        pred_cy = box_preds[bi, row, col, b, 1]
                        pred_w = box_preds[bi, row, col, b, 2]
                        pred_h = box_preds[bi, row, col, b, 3]
                        pred_conf = box_preds[bi, row, col, b, 4]

                        if (row, col, b) in assigned:
                            cx_m, cy_m, w_m, h_m, iou_val, cls_m = assigned[
                                (row, col, b)
                            ]

                            # Target cx/cy in cell-relative sigmoid-space
                            tgt_cx_rel = cx_m / cell_w_px - float(col)
                            tgt_cy_rel = cy_m / cell_h_px - float(row)
                            # Clamp to valid sigmoid output range (avoid NaN in log)
                            tgt_cx_rel = max(1e-6, min(1.0 - 1e-6, tgt_cx_rel))
                            tgt_cy_rel = max(1e-6, min(1.0 - 1e-6, tgt_cy_rel))

                            # Coordinate loss: MSE on sigmoid outputs
                            sig_cx = F.sigmoid(pred_cx)
                            sig_cy = F.sigmoid(pred_cy)
                            tgt_cx_t = lucid.tensor([tgt_cx_rel])
                            tgt_cy_t = lucid.tensor([tgt_cy_rel])

                            xy_terms.append((sig_cx - tgt_cx_t[0]) ** 2)
                            xy_terms.append((sig_cy - tgt_cy_t[0]) ** 2)

                            # wh loss: MSE on sqrt of (exp(pred) * img_dim)
                            pred_w_px = lucid.exp(
                                pred_w.clamp(min=-10.0, max=10.0)
                            ) * float(W)
                            pred_h_px = lucid.exp(
                                pred_h.clamp(min=-10.0, max=10.0)
                            ) * float(H)
                            tgt_sqrt_w = float(math.sqrt(max(w_m, 0.0)))
                            tgt_sqrt_h = float(math.sqrt(max(h_m, 0.0)))
                            sqrt_pw = lucid.log(pred_w_px.clamp(min=1e-6)) * 0.5
                            sqrt_ph = lucid.log(pred_h_px.clamp(min=1e-6)) * 0.5
                            # Approximate sqrt MSE as (sqrt(pw) - sqrt(tw))²
                            # but we can also do it directly:
                            wh_terms.append(
                                (pred_w_px**0.5 - lucid.tensor([tgt_sqrt_w**2]) ** 0.5)
                                ** 2
                            )
                            wh_terms.append(
                                (pred_h_px**0.5 - lucid.tensor([tgt_sqrt_h**2]) ** 0.5)
                                ** 2
                            )
                            _ = sqrt_pw  # suppress unused
                            _ = sqrt_ph

                            # Confidence loss (obj): MSE vs iou_val
                            sig_conf = F.sigmoid(pred_conf)
                            tgt_iou_t = lucid.tensor([iou_val])
                            conf_obj.append((sig_conf - tgt_iou_t[0]) ** 2)

                            # Class loss: MSE on class scores vs one-hot
                            pred_cls = cls_preds[bi, row, col, :]  # (C,)
                            tgt_cls_list = [0.0] * C
                            tgt_cls_list[cls_m] = 1.0
                            tgt_cls_t = lucid.tensor(tgt_cls_list)
                            cls_terms.append(((pred_cls - tgt_cls_t) ** 2).sum())

                        else:
                            # No-object: confidence should be 0
                            sig_conf = F.sigmoid(pred_conf)
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
        image_size: tuple[int, int] = (448, 448),
    ) -> ObjectDetectionOutput:
        """Run YOLOv1 forward pass.

        Args:
            x:          (B, C, H, W) input image tensor.
            targets:    Optional list of B dicts for training loss computation.
                        Each dict has:
                        - ``"boxes"``:  (M_i, 4) xyxy boxes normalised to [0, 1].
                        - ``"labels"``: (M_i,) integer class indices.
            image_size: (H, W) of the input in pixels (used for box decoding).
                        Defaults to 448×448 (the original paper resolution).

        Returns:
            :class:`~lucid.models._output.ObjectDetectionOutput` with:
            - ``logits``:     (B, S·S·num_boxes, num_classes)
            - ``pred_boxes``: (B, S·S·num_boxes, 4) xyxy pixel boxes
            - ``loss``:       scalar tensor when targets are provided, else None
        """
        raw = self._forward_features(x)
        logits, pred_boxes = self._decode_predictions(raw, image_size)

        loss: Tensor | None = None
        if targets is not None:
            loss = self._compute_loss(raw, targets, image_size)

        return ObjectDetectionOutput(logits=logits, pred_boxes=pred_boxes, loss=loss)

    def postprocess(
        self,
        output: ObjectDetectionOutput,
        image_size: tuple[int, int] = (448, 448),
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
            lg_b = output.logits[b]  # (S*S*B, C)
            bx_b = output.pred_boxes[b]  # (S*S*B, 4)
            # Confidence scores: max class prob (no sigmoid — raw logits)
            sc_b = F.sigmoid(lg_b)  # (S*S*B, C)

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

_CFG_V1 = YOLOV1Config(
    num_classes=80,
    in_channels=3,
    split_size=7,
    num_boxes=2,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    score_thresh=0.5,
    nms_thresh=0.5,
    tiny=False,
)

_CFG_V1_TINY = YOLOV1Config(
    num_classes=80,
    in_channels=3,
    split_size=7,
    num_boxes=2,
    lambda_coord=5.0,
    lambda_noobj=0.5,
    score_thresh=0.5,
    nms_thresh=0.5,
    tiny=True,
)


def _make_v1(
    cfg: YOLOV1Config, overrides: dict[str, object]
) -> YOLOV1ForObjectDetection:
    if overrides:
        merged = {**cfg.__dict__, **overrides}
        merged.pop("model_type", None)
        cfg = YOLOV1Config(**merged)
    return YOLOV1ForObjectDetection(cfg)


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v1",
    model_class=YOLOV1ForObjectDetection,
    default_config=_CFG_V1,
)
def yolo_v1(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOV1ForObjectDetection:
    """YOLOv1 with full Darknet backbone (Redmon et al., CVPR 2016).

    Single-shot detector that divides the image into a 7×7 grid and predicts
    2 bounding boxes + 80 class scores per cell in one forward pass.
    """
    return _make_v1(_CFG_V1, overrides)


@register_model(
    task="object-detection",
    family="yolo",
    model_type="yolo_v1",
    model_class=YOLOV1ForObjectDetection,
    default_config=_CFG_V1_TINY,
)
def yolo_v1_tiny(
    pretrained: bool = False,
    **overrides: object,
) -> YOLOV1ForObjectDetection:
    """YOLOv1 with tiny Darknet backbone — lightweight variant.

    Uses a smaller convolutional backbone for faster inference at the cost
    of detection accuracy.
    """
    return _make_v1(_CFG_V1_TINY, overrides)
