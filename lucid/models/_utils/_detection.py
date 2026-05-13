"""Detection-task utilities: box ops, NMS, anchors, RoI ops, and shared modules.

Structure
---------
§1  Box operations   — pure functions on (N, 4) xyxy Tensors
§2  NMS              — greedy non-maximum suppression
§3  Anchor generator — multi-scale, multi-ratio anchor boxes
§4  RoI operations   — RoI Align and RoI Pool
§5  Shared nn.Module — FPN, RPN, RoI head (reused across RCNN family)
"""

import math
from typing import cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor

# ---------------------------------------------------------------------------
# §1  Box operations
# ---------------------------------------------------------------------------


def box_area(boxes: Tensor) -> Tensor:
    """Area of boxes in xyxy format.

    Args:
        boxes: (N, 4) — each row is [x1, y1, x2, y2].

    Returns:
        (N,) area tensor.
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise IoU matrix between two sets of boxes (xyxy format).

    Args:
        boxes1: (N, 4)
        boxes2: (M, 4)

    Returns:
        (N, M) IoU matrix.
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    # Broadcast for pairwise intersection: (N, 1, 4) vs (1, M, 4)
    b1 = boxes1[:, None, :]  # (N, 1, 4)
    b2 = boxes2[None, :, :]  # (1, M, 4)

    # Element-wise max/min to get intersection corners
    inter_x1: Tensor = lucid.maximum(b1[..., 0], b2[..., 0])  # (N, M)
    inter_y1: Tensor = lucid.maximum(b1[..., 1], b2[..., 1])
    inter_x2: Tensor = lucid.minimum(b1[..., 2], b2[..., 2])
    inter_y2: Tensor = lucid.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h  # (N, M)

    union = area1[:, None] + area2[None, :] - inter_area
    return inter_area / union.clamp(min=1e-6)


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """Pairwise Generalised IoU (GIoU) — same shape convention as ``box_iou``.

    GIoU = IoU - |C \\ (A ∪ B)| / |C|  where C is the smallest enclosing box.

    Args:
        boxes1: (N, 4) xyxy
        boxes2: (M, 4) xyxy

    Returns:
        (N, M) GIoU matrix in [-1, 1].
    """
    area1 = box_area(boxes1)  # (N,)
    area2 = box_area(boxes2)  # (M,)

    b1 = boxes1[:, None, :]  # (N, 1, 4)
    b2 = boxes2[None, :, :]  # (1, M, 4)

    # Intersection
    inter_x1: Tensor = lucid.maximum(b1[..., 0], b2[..., 0])
    inter_y1: Tensor = lucid.maximum(b1[..., 1], b2[..., 1])
    inter_x2: Tensor = lucid.minimum(b1[..., 2], b2[..., 2])
    inter_y2: Tensor = lucid.minimum(b1[..., 3], b2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0.0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0.0)
    inter_area = inter_w * inter_h  # (N, M)

    union = area1[:, None] + area2[None, :] - inter_area

    # Enclosing box
    enc_x1: Tensor = lucid.minimum(b1[..., 0], b2[..., 0])
    enc_y1: Tensor = lucid.minimum(b1[..., 1], b2[..., 1])
    enc_x2: Tensor = lucid.maximum(b1[..., 2], b2[..., 2])
    enc_y2: Tensor = lucid.maximum(b1[..., 3], b2[..., 3])

    enc_area = (enc_x2 - enc_x1).clamp(min=0.0) * (enc_y2 - enc_y1).clamp(min=0.0)

    iou = inter_area / union.clamp(min=1e-6)
    return iou - (enc_area - union) / enc_area.clamp(min=1e-6)


def box_xyxy_to_cxcywh(boxes: Tensor) -> Tensor:
    """Convert (x1, y1, x2, y2) → (cx, cy, w, h).

    Args:
        boxes: (..., 4) in xyxy format.

    Returns:
        (..., 4) in cxcywh format.
    """
    x1 = boxes[..., 0:1]
    y1 = boxes[..., 1:2]
    x2 = boxes[..., 2:3]
    y2 = boxes[..., 3:4]
    return lucid.cat([(x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1], dim=-1)


def box_cxcywh_to_xyxy(boxes: Tensor) -> Tensor:
    """Convert (cx, cy, w, h) → (x1, y1, x2, y2).

    Args:
        boxes: (..., 4) in cxcywh format.

    Returns:
        (..., 4) in xyxy format.
    """
    cx = boxes[..., 0:1]
    cy = boxes[..., 1:2]
    w = boxes[..., 2:3]
    h = boxes[..., 3:4]
    return lucid.cat([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], dim=-1)


def clip_boxes_to_image(boxes: Tensor, size: tuple[int, int]) -> Tensor:
    """Clip boxes to image boundaries.

    Args:
        boxes: (N, 4) xyxy in pixel coordinates.
        size:  (height, width) of the image.

    Returns:
        (N, 4) clipped boxes.
    """
    h, w = size
    x1 = boxes[:, 0:1].clamp(min=0.0, max=float(w))
    y1 = boxes[:, 1:2].clamp(min=0.0, max=float(h))
    x2 = boxes[:, 2:3].clamp(min=0.0, max=float(w))
    y2 = boxes[:, 3:4].clamp(min=0.0, max=float(h))
    return lucid.cat([x1, y1, x2, y2], dim=1)


def remove_small_boxes(boxes: Tensor, min_size: float) -> Tensor:
    """Return indices of boxes whose width AND height are ≥ min_size.

    Args:
        boxes:    (N, 4) xyxy.
        min_size: Minimum side length in pixels.

    Returns:
        1-D index Tensor of surviving boxes.
    """
    ws = boxes[:, 2] - boxes[:, 0]
    hs = boxes[:, 3] - boxes[:, 1]
    keep: list[int] = [
        i
        for i in range(int(boxes.shape[0]))
        if float(ws[i].item()) >= min_size and float(hs[i].item()) >= min_size
    ]
    if not keep:
        return lucid.zeros((0,))
    return lucid.tensor(keep)


def encode_boxes(
    reference_boxes: Tensor,
    proposals: Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> Tensor:
    """Encode box regression targets (dx, dy, dw, dh).

    Args:
        reference_boxes: (N, 4) xyxy ground-truth boxes.
        proposals:       (N, 4) xyxy anchor / proposal boxes.
        weights:         Per-component scaling (wx, wy, ww, wh).

    Returns:
        (N, 4) regression targets.
    """
    wx, wy, ww, wh = weights

    ref = box_xyxy_to_cxcywh(reference_boxes)
    pro = box_xyxy_to_cxcywh(proposals)

    dx = wx * (ref[:, 0] - pro[:, 0]) / pro[:, 2].clamp(min=1e-6)
    dy = wy * (ref[:, 1] - pro[:, 1]) / pro[:, 3].clamp(min=1e-6)
    dw = ww * lucid.log(ref[:, 2] / pro[:, 2].clamp(min=1e-6))
    dh = wh * lucid.log(ref[:, 3] / pro[:, 3].clamp(min=1e-6))

    return lucid.stack([dx, dy, dw, dh], dim=1)


def decode_boxes(
    deltas: Tensor,
    anchors: Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    bbox_xform_clip: float = math.log(1000.0 / 16),
) -> Tensor:
    """Decode box regression deltas back to xyxy format.

    Args:
        deltas:           (N, 4) regression outputs.
        anchors:          (N, 4) xyxy reference boxes.
        weights:          Must match ``encode_boxes``.
        bbox_xform_clip:  Clamps dw/dh to prevent exp overflow.

    Returns:
        (N, 4) decoded boxes in xyxy format.
    """
    wx, wy, ww, wh = weights

    anc = box_xyxy_to_cxcywh(anchors)
    acx = anc[:, 0]
    acy = anc[:, 1]
    aw = anc[:, 2]
    ah = anc[:, 3]

    dx = deltas[:, 0] / wx
    dy = deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=bbox_xform_clip)
    dh = (deltas[:, 3] / wh).clamp(max=bbox_xform_clip)

    pred_cx = dx * aw + acx
    pred_cy = dy * ah + acy
    pred_w = lucid.exp(dw) * aw
    pred_h = lucid.exp(dh) * ah

    x1 = pred_cx - pred_w / 2.0
    y1 = pred_cy - pred_h / 2.0
    x2 = pred_cx + pred_w / 2.0
    y2 = pred_cy + pred_h / 2.0

    return lucid.stack([x1, y1, x2, y2], dim=1)


# ---------------------------------------------------------------------------
# §2  Non-maximum suppression
# ---------------------------------------------------------------------------


def nms(
    boxes: Tensor,
    scores: Tensor,
    iou_threshold: float,
) -> Tensor:
    """Greedy NMS — returns indices of surviving boxes, sorted by score desc.

    Algorithm:
      1. Sort boxes by descending score.
      2. Iterate: keep highest-scoring box; suppress remaining boxes with
         IoU > ``iou_threshold`` against the kept box.

    The loop is O(N²) but operates on scalars extracted via ``.item()``,
    which is acceptable for typical proposal counts (≤ a few thousand).

    Args:
        boxes:         (N, 4) xyxy float Tensor.
        scores:        (N,) confidence scores.
        iou_threshold: Suppress if IoU > this value.

    Returns:
        1-D int Tensor of surviving box indices (descending score order).
    """
    N: int = int(boxes.shape[0])
    if N == 0:
        return lucid.zeros((0,))

    # Sort descending by score using Python-level argsort on negated scores
    order: list[int] = sorted(
        range(N), key=lambda i: float(scores[i].item()), reverse=True
    )

    suppressed: list[bool] = [False] * N
    keep: list[int] = []

    for i in range(len(order)):
        idx = order[i]
        if suppressed[idx]:
            continue
        keep.append(idx)
        box_i = boxes[idx : idx + 1]  # (1, 4)
        for j in range(i + 1, len(order)):
            jdx = order[j]
            if suppressed[jdx]:
                continue
            box_j = boxes[jdx : jdx + 1]  # (1, 4)
            iou_val = float(box_iou(box_i, box_j)[0, 0].item())
            if iou_val > iou_threshold:
                suppressed[jdx] = True

    if not keep:
        return lucid.zeros((0,))
    return lucid.tensor(keep)


def batched_nms(
    boxes: Tensor,
    scores: Tensor,
    idxs: Tensor,
    iou_threshold: float,
) -> Tensor:
    """NMS applied independently per class via the class-offset trick.

    Boxes from different classes are offset by a large class-dependent
    value, preventing cross-class suppression in a single NMS pass.

    Args:
        boxes:         (N, 4) xyxy.
        scores:        (N,) confidence scores.
        idxs:          (N,) integer class index per box.
        iou_threshold: IoU threshold.

    Returns:
        Surviving box indices (sorted by score, descending).
    """
    max_coord = float(boxes.max().item())
    offsets = idxs.float() * (max_coord + 1.0)
    boxes_for_nms = boxes + offsets[:, None]
    return nms(boxes_for_nms, scores, iou_threshold)


# ---------------------------------------------------------------------------
# §3  Anchor generator
# ---------------------------------------------------------------------------


class AnchorGenerator(nn.Module):
    """Generate multi-scale, multi-ratio anchors for each FPN level.

    For each (feature map, stride) pair the generator produces anchors
    centred at each spatial cell.

    Args:
        sizes:         Anchor sizes (sqrt of area) per FPN level,
                       e.g. ``((32,), (64,), (128,), (256,), (512,))``.
        aspect_ratios: Width/height ratios per FPN level,
                       e.g. ``((0.5, 1.0, 2.0),) * 5``.
    """

    def __init__(
        self,
        sizes: tuple[tuple[int, ...], ...] = ((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios: tuple[tuple[float, ...], ...] = ((0.5, 1.0, 2.0),) * 5,
    ) -> None:
        super().__init__()
        assert len(sizes) == len(
            aspect_ratios
        ), "sizes and aspect_ratios must have the same number of levels"
        self.sizes = sizes
        self.aspect_ratios = aspect_ratios
        self._cell_anchors: list[Tensor] = self._compute_cell_anchors()

    def _compute_cell_anchors(self) -> list[Tensor]:
        """Pre-compute base anchors (centred at origin) for every FPN level."""
        all_anchors: list[Tensor] = []
        for level_sizes, level_ratios in zip(self.sizes, self.aspect_ratios):
            anchors: list[list[float]] = []
            for size in level_sizes:
                area = float(size * size)
                for ratio in level_ratios:
                    w = math.sqrt(area / ratio)
                    h = w * ratio
                    anchors.append([-w / 2.0, -h / 2.0, w / 2.0, h / 2.0])
            all_anchors.append(lucid.tensor(anchors))
        return all_anchors

    def _grid_anchors(
        self,
        feature_map_size: tuple[int, int],
        stride: tuple[int, int],
        base_anchors: Tensor,
    ) -> Tensor:
        """Tile base_anchors across a feature map grid.

        Args:
            feature_map_size: (H, W) of the feature map.
            stride:           (stride_h, stride_w) pixels per cell.
            base_anchors:     (A, 4) base anchors centred at origin.

        Returns:
            (H × W × A, 4) anchors in xyxy image-pixel coordinates.
        """
        fH, fW = feature_map_size
        sH, sW = stride

        # Build shift table: (fH*fW, 4) — (cx, cy, cx, cy)
        shifts: list[list[float]] = [
            [(c + 0.5) * sW, (r + 0.5) * sH, (c + 0.5) * sW, (r + 0.5) * sH]
            for r in range(fH)
            for c in range(fW)
        ]
        shifts_t = lucid.tensor(shifts)  # (G, 4)

        G = fH * fW
        A = int(base_anchors.shape[0])

        # (G, 1, 4) + (1, A, 4) → (G, A, 4) → (G*A, 4)
        grid = shifts_t[:, None, :] + base_anchors[None, :, :]
        return grid.reshape(G * A, 4)

    def forward(  # type: ignore[override]
        self,
        feature_maps: list[Tensor],
        image_size: tuple[int, int],
        strides: list[tuple[int, int]],
    ) -> list[Tensor]:
        """Generate anchors for all FPN levels.

        Args:
            feature_maps: One (B, C, H, W) tensor per FPN level.
            image_size:   (H, W) of the input image (unused here, kept for
                          API symmetry with clip_boxes callers).
            strides:      (stride_h, stride_w) per level.

        Returns:
            List of (H_l × W_l × A_l, 4) anchor tensors, one per level.
        """
        assert len(feature_maps) == len(self._cell_anchors)
        all_anchors: list[Tensor] = []
        for feat, base, stride in zip(feature_maps, self._cell_anchors, strides):
            fH = int(feat.shape[2])
            fW = int(feat.shape[3])
            all_anchors.append(self._grid_anchors((fH, fW), stride, base))
        return all_anchors


# ---------------------------------------------------------------------------
# §4  RoI operations
# ---------------------------------------------------------------------------


def roi_align(
    input: Tensor,
    boxes: list[Tensor],
    output_size: int | tuple[int, int],
    spatial_scale: float = 1.0,
    aligned: bool = True,
) -> Tensor:
    """RoI Align — bilinear sub-pixel sampling into fixed-size crops.

    Uses ``F.grid_sample`` (bilinear) to sample the feature map at the
    exact continuous coordinates of each RoI, replicating the reference
    implementation in Mask R-CNN.

    Args:
        input:        (B, C, H, W) feature map.
        boxes:        List of B tensors, each (N_i, 4) xyxy in *image*
                      pixel coordinates.
        output_size:  Height and width of each RoI crop.
        spatial_scale: Ratio of feature-map size to input image size
                       (e.g. 1/32 for a stride-32 backbone level).
        aligned:      When True, apply the 0.5-pixel alignment offset.

    Returns:
        (sum(N_i), C, out_h, out_w) stacked crops.
    """
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    feat_H = int(input.shape[2])
    feat_W = int(input.shape[3])
    C = int(input.shape[1])

    results: list[Tensor] = []

    for b_idx, roi_boxes in enumerate(boxes):
        N = int(roi_boxes.shape[0])
        if N == 0:
            continue

        feat: Tensor = input[b_idx : b_idx + 1]  # (1, C, H, W)

        x1 = roi_boxes[:, 0] * spatial_scale
        y1 = roi_boxes[:, 1] * spatial_scale
        x2 = roi_boxes[:, 2] * spatial_scale
        y2 = roi_boxes[:, 3] * spatial_scale

        if aligned:
            x1 = x1 - 0.5
            y1 = y1 - 0.5
            x2 = x2 - 0.5
            y2 = y2 - 0.5

        # Build (N, out_h, out_w, 2) sampling grid using Python loops
        grid_data: list[list[list[list[float]]]] = []
        for n in range(N):
            rx1 = float(x1[n].item())
            rx2 = float(x2[n].item())
            ry1 = float(y1[n].item())
            ry2 = float(y2[n].item())

            # Sample centres: out_w points in [rx1, rx2], out_h in [ry1, ry2]
            xs = [rx1 + (rx2 - rx1) * (i + 0.5) / out_w for i in range(out_w)]
            ys = [ry1 + (ry2 - ry1) * (j + 0.5) / out_h for j in range(out_h)]

            # Normalise to [-1, 1] (grid_sample convention)
            norm_xs = [xi / (feat_W - 1) * 2.0 - 1.0 for xi in xs]
            norm_ys = [yi / (feat_H - 1) * 2.0 - 1.0 for yi in ys]

            row: list[list[list[float]]] = []
            for j in range(out_h):
                col: list[list[float]] = [
                    [norm_xs[i], norm_ys[j]] for i in range(out_w)
                ]
                row.append(col)
            grid_data.append(row)

        grid = lucid.tensor(grid_data)  # (N, H, W, 2)

        feat_n = feat.expand(N, -1, -1, -1)
        sampled = F.grid_sample(feat_n, grid, mode="bilinear", align_corners=True)
        results.append(sampled)

    if not results:
        return lucid.zeros((0, C, out_h, out_w))
    return lucid.cat(results, dim=0)


def roi_pool(
    input: Tensor,
    boxes: list[Tensor],
    output_size: int | tuple[int, int],
    spatial_scale: float = 1.0,
) -> Tensor:
    """RoI Pool (max-pool variant used in R-CNN / Fast R-CNN).

    Quantises RoI boundaries to integer pixels then adaptively
    average-pools each bin to ``output_size``.

    Args:
        input:        (B, C, H, W) feature map.
        boxes:        List of B tensors, each (N_i, 4) xyxy image coords.
        output_size:  (out_h, out_w) of each crop.
        spatial_scale: Feature-map to image scale ratio.

    Returns:
        (sum(N_i), C, out_h, out_w).
    """
    if isinstance(output_size, int):
        out_h = out_w = output_size
    else:
        out_h, out_w = output_size

    feat_H = int(input.shape[2])
    feat_W = int(input.shape[3])
    C = int(input.shape[1])

    pool = nn.AdaptiveAvgPool2d((out_h, out_w))
    results: list[Tensor] = []

    for b_idx, roi_boxes in enumerate(boxes):
        N = int(roi_boxes.shape[0])
        if N == 0:
            continue
        feat: Tensor = input[b_idx]  # (C, H, W)

        for n in range(N):
            x1 = int(round(float(roi_boxes[n, 0].item()) * spatial_scale))
            y1 = int(round(float(roi_boxes[n, 1].item()) * spatial_scale))
            x2 = int(round(float(roi_boxes[n, 2].item()) * spatial_scale))
            y2 = int(round(float(roi_boxes[n, 3].item()) * spatial_scale))

            x1 = max(0, min(x1, feat_W - 1))
            y1 = max(0, min(y1, feat_H - 1))
            x2 = max(x1 + 1, min(x2, feat_W))
            y2 = max(y1 + 1, min(y2, feat_H))

            crop: Tensor = feat[:, y1:y2, x1:x2]  # (C, rH, rW)
            pooled = cast(Tensor, pool(crop.unsqueeze(0))).squeeze(0)
            results.append(pooled.unsqueeze(0))

    if not results:
        return lucid.zeros((0, C, out_h, out_w))
    return lucid.cat(results, dim=0)


# ---------------------------------------------------------------------------
# §5  Shared nn.Module components
# ---------------------------------------------------------------------------


class FPN(nn.Module):
    """Feature Pyramid Network (Lin et al., 2017).

    Merges multi-scale backbone feature maps into a unified pyramid of
    semantically rich, spatially precise levels.

    Architecture per level:
      lateral  : Conv2d(in_ch, out_ch, 1)
      output   : Conv2d(out_ch, out_ch, 3, padding=1)
      top-down : upsample(2×, nearest) + element-wise add

    Args:
        in_channels:  Channel counts of each bottom-up map, finest first
                      (e.g. [256, 512, 1024, 2048] for ResNet C2–C5).
        out_channels: Unified channel count for all pyramid levels.
        extra_blocks: Additional coarser levels appended via 3×3 stride-2
                      conv on the coarsest FPN output (default: 1 → P6).
    """

    def __init__(
        self,
        in_channels: list[int],
        out_channels: int,
        extra_blocks: int = 1,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        n = len(in_channels)

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ic, out_channels, 1) for ic in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in range(n)]
        )
        self.extra_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                for _ in range(extra_blocks)
            ]
        )

    def forward(self, features: list[Tensor]) -> list[Tensor]:  # type: ignore[override]
        """
        Args:
            features: Bottom-up maps, finest → coarsest (e.g. C2, C3, C4, C5).

        Returns:
            FPN outputs, finest → coarsest (e.g. P2, P3, P4, P5, P6).
        """
        # Lateral projections
        laterals: list[Tensor] = [
            cast(Tensor, lat(f)) for lat, f in zip(self.lateral_convs, features)
        ]

        # Top-down: merge from coarsest to finest
        n = len(laterals)
        for i in range(n - 2, -1, -1):
            up = F.interpolate(laterals[i + 1], scale_factor=2.0, mode="nearest")
            laterals[i] = laterals[i] + up

        # 3×3 output convolutions (anti-aliasing)
        outs: list[Tensor] = [
            cast(Tensor, conv(lat)) for conv, lat in zip(self.output_convs, laterals)
        ]

        # Extra coarser levels
        extra_in = outs[-1]
        for conv in self.extra_convs:
            extra_in = F.relu(cast(Tensor, conv(extra_in)))
            outs.append(extra_in)

        return outs


class RPN(nn.Module):
    """Region Proposal Network (Ren et al., 2015).

    Slides a 3×3 conv over each FPN level to predict per-anchor:
      - objectness score  (foreground vs background)
      - box delta         (dx, dy, dw, dh)

    Proposals from all levels are merged, clipped, filtered and NMS'd
    per image to produce the final region proposals.

    Args:
        in_channels:    FPN output channels.
        num_anchors:    Anchors per spatial cell (len(sizes) × len(ratios)).
        pre_nms_top_n:  Proposals kept per level before NMS.
        post_nms_top_n: Proposals kept per image after NMS.
        nms_threshold:  IoU threshold for NMS.
        min_size:       Minimum proposal side length (pixels).
        score_thresh:   Minimum objectness score (post-sigmoid).
    """

    def __init__(
        self,
        in_channels: int,
        num_anchors: int,
        pre_nms_top_n: int = 2000,
        post_nms_top_n: int = 1000,
        nms_threshold: float = 0.7,
        min_size: float = 1.0,
        score_thresh: float = 0.0,
    ) -> None:
        super().__init__()
        self.pre_nms_top_n = pre_nms_top_n
        self.post_nms_top_n = post_nms_top_n
        self.nms_threshold = nms_threshold
        self.min_size = min_size
        self.score_thresh = score_thresh

        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)

    def forward(  # type: ignore[override]
        self,
        features: list[Tensor],
        anchors: list[Tensor],
        image_size: tuple[int, int],
    ) -> tuple[list[Tensor], list[Tensor]]:
        """Run RPN over all FPN levels and return proposals per image.

        Args:
            features:   FPN outputs, each (B, C, H_l, W_l).
            anchors:    Per-level anchor tensors from AnchorGenerator.
            image_size: (H, W) of the input image.

        Returns:
            (proposals, scores):
                proposals[b]: (K_b, 4) xyxy proposals for image b.
                scores[b]:    (K_b,)   objectness probabilities.
        """
        B = int(features[0].shape[0])
        all_proposals: list[list[Tensor]] = [[] for _ in range(B)]
        all_scores: list[list[Tensor]] = [[] for _ in range(B)]

        for feat, level_anchors in zip(features, anchors):
            t = F.relu(cast(Tensor, self.conv(feat)))
            logits = cast(Tensor, self.cls_logits(t))  # (B, A, H, W)
            deltas = cast(Tensor, self.bbox_pred(t))  # (B, 4A, H, W)

            A = int(logits.shape[1])
            fH = int(logits.shape[2])
            fW = int(logits.shape[3])

            scores_flat = F.sigmoid(logits.reshape(B, -1))  # (B, A*H*W)
            deltas_flat = deltas.reshape(B, A * fH * fW, 4)  # (B, A*H*W, 4)

            for b in range(B):
                sc = scores_flat[b]  # (N_anc,)
                dl = deltas_flat[b]  # (N_anc, 4)

                K = min(self.pre_nms_top_n, int(sc.shape[0]))
                # argsort ascending on negated scores → top-K indices
                topk_idx = lucid.argsort(-sc)[:K]

                topk_sc = sc[topk_idx]
                topk_dl = dl[topk_idx]
                topk_anc = level_anchors[topk_idx]

                props = decode_boxes(topk_dl, topk_anc)
                props = clip_boxes_to_image(props, image_size)

                keep_small = remove_small_boxes(props, self.min_size)
                if int(keep_small.shape[0]) == 0:
                    continue

                props = props[keep_small]
                topk_sc = topk_sc[keep_small]

                score_mask: list[int] = [
                    i
                    for i in range(int(props.shape[0]))
                    if float(topk_sc[i].item()) >= self.score_thresh
                ]
                if not score_mask:
                    continue

                mask_t = lucid.tensor(score_mask)
                props = props[mask_t]
                topk_sc = topk_sc[mask_t]

                all_proposals[b].append(props)
                all_scores[b].append(topk_sc)

        final_proposals: list[Tensor] = []
        final_scores: list[Tensor] = []

        for b in range(B):
            if not all_proposals[b]:
                final_proposals.append(lucid.zeros((0, 4)))
                final_scores.append(lucid.zeros((0,)))
                continue

            props_b = lucid.cat(all_proposals[b], dim=0)
            sc_b = lucid.cat(all_scores[b], dim=0)

            keep = nms(props_b, sc_b, self.nms_threshold)
            K2 = min(self.post_nms_top_n, int(keep.shape[0]))
            keep = keep[:K2]

            final_proposals.append(props_b[keep])
            final_scores.append(sc_b[keep])

        return final_proposals, final_scores


class RoIHead(nn.Module):
    """Two-FC RoI head shared by Fast R-CNN, Faster R-CNN and Mask R-CNN.

    Takes RoI-aligned crops and predicts class logits and box deltas.

    Args:
        in_channels:         Channels of each RoI crop.
        roi_size:            (H, W) of the RoI Align output crop.
        num_classes:         Foreground classes (background adds +1).
        representation_size: Hidden size of the two FC layers.
    """

    def __init__(
        self,
        in_channels: int,
        roi_size: int | tuple[int, int],
        num_classes: int,
        representation_size: int = 1024,
    ) -> None:
        super().__init__()
        if isinstance(roi_size, int):
            roi_h = roi_w = roi_size
        else:
            roi_h, roi_w = roi_size

        flat_size = in_channels * roi_h * roi_w

        self.fc6 = nn.Linear(flat_size, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.cls_score = nn.Linear(representation_size, num_classes + 1)
        self.bbox_pred = nn.Linear(representation_size, num_classes * 4)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        """
        Args:
            x: (N_rois, C, roi_h, roi_w) RoI-aligned feature crops.

        Returns:
            (class_logits, box_deltas):
                class_logits: (N_rois, num_classes + 1)
                box_deltas:   (N_rois, num_classes * 4)
        """
        x = x.flatten(1)
        x = F.relu(cast(Tensor, self.fc6(x)))
        x = F.relu(cast(Tensor, self.fc7(x)))
        class_logits = cast(Tensor, self.cls_score(x))
        box_deltas = cast(Tensor, self.bbox_pred(x))
        return class_logits, box_deltas
