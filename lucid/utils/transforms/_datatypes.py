"""Typed targets for multi-target transforms (torchvision-v2 style).

A transform applied to a *sample* (image + companions) must move every
target consistently: a flip mirrors the image, its segmentation mask,
and its bounding boxes together.  To dispatch correctly, companions are
wrapped in lightweight typed objects:

* :class:`Image`        — the pixel image (a plain ``Tensor`` is also
  treated as an image, so single-image pipelines need no wrapping).
* :class:`Mask`         — a segmentation / label map; geometric
  transforms resample it with *nearest* interpolation, photometric
  transforms leave it untouched.
* :class:`BoundingBoxes`— ``(N, 4)`` boxes in a named ``format`` with a
  ``canvas_size``; geometric transforms update the coordinates.

Wrappers hold a :class:`lucid.Tensor` in ``.data`` plus metadata; they
are intentionally not ``Tensor`` subclasses (Lucid's tensor is
C-backed) — transforms dispatch on ``isinstance``.
"""

from dataclasses import dataclass

import lucid
from lucid._tensor import Tensor

# Supported bounding-box coordinate formats.
_BOX_FORMATS = frozenset({"xyxy", "xywh", "cxcywh"})


def _cat(tensors: list[Tensor], dim: int) -> Tensor:
    """``lucid.concat`` wrapper (absorbs the int32-typed ``dim`` stub)."""
    return lucid.concat(tensors, dim=dim)  # type: ignore[arg-type]


@dataclass
class Image:
    """Wraps a pixel image tensor ``(C, H, W)`` or ``(B, C, H, W)``."""

    data: Tensor


@dataclass
class Mask:
    """Wraps a segmentation / label mask, resampled with nearest mode."""

    data: Tensor


@dataclass
class Keypoints:
    """Wraps ``(N, D)`` keypoints (``D >= 2``; first two columns ``x, y``).

    Extra columns (e.g. visibility / angle / scale) are carried through
    geometric transforms unchanged; only the ``x, y`` coordinates move.

    Parameters
    ----------
    data : Tensor
        ``(N, D)`` with ``D >= 2``; columns 0 and 1 are ``x`` and ``y``.
    canvas_size : (int, int)
        ``(H, W)`` of the image the points index into.
    """

    data: Tensor
    canvas_size: tuple[int, int]


@dataclass
class BoundingBoxes:
    """Wraps ``(N, 4)`` bounding boxes with a format + canvas size.

    Parameters
    ----------
    data : Tensor
        ``(N, 4)`` coordinates in ``format``.
    format : {"xyxy", "xywh", "cxcywh"}
        Coordinate convention: corner-corner, corner-size, or
        center-size.
    canvas_size : (int, int)
        ``(H, W)`` of the image the boxes index into.  Updated by
        geometric transforms (crop / resize / pad).
    """

    data: Tensor
    format: str
    canvas_size: tuple[int, int]

    def __post_init__(self) -> None:
        if self.format not in _BOX_FORMATS:
            raise ValueError(
                f"BoundingBoxes: unknown format {self.format!r}; "
                f"expected one of {sorted(_BOX_FORMATS)}"
            )


def to_xyxy(boxes: BoundingBoxes) -> Tensor:
    """Return the box coordinates as ``(N, 4)`` xyxy regardless of format."""
    d = boxes.data
    if boxes.format == "xyxy":
        return d
    a = d[:, 0:1]
    b = d[:, 1:2]
    c = d[:, 2:3]
    e = d[:, 3:4]
    if boxes.format == "xywh":
        # (x, y, w, h) -> (x, y, x+w, y+h)
        return _cat([a, b, a + c, b + e], 1)
    # cxcywh: (cx, cy, w, h) -> corners
    return _cat([a - c / 2, b - e / 2, a + c / 2, b + e / 2], 1)


def from_xyxy(xyxy: Tensor, fmt: str) -> Tensor:
    """Convert ``(N, 4)`` xyxy coordinates back to ``fmt``."""
    if fmt == "xyxy":
        return xyxy
    x1 = xyxy[:, 0:1]
    y1 = xyxy[:, 1:2]
    x2 = xyxy[:, 2:3]
    y2 = xyxy[:, 3:4]
    if fmt == "xywh":
        return _cat([x1, y1, x2 - x1, y2 - y1], 1)
    # cxcywh
    return _cat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)


def _rebuild(
    boxes: BoundingBoxes, xyxy: Tensor, canvas: tuple[int, int]
) -> BoundingBoxes:
    """Re-wrap transformed xyxy coords back into the original format."""
    return BoundingBoxes(from_xyxy(xyxy, boxes.format), boxes.format, canvas)


def flip_boxes(boxes: BoundingBoxes, *, horizontal: bool) -> BoundingBoxes:
    """Mirror boxes horizontally or vertically within the canvas."""
    h, w = boxes.canvas_size
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    if horizontal:
        new = _cat([w - x2, y1, w - x1, y2], 1)
    else:
        new = _cat([x1, h - y2, x2, h - y1], 1)
    return _rebuild(boxes, new, boxes.canvas_size)


def resize_boxes(boxes: BoundingBoxes, new_h: int, new_w: int) -> BoundingBoxes:
    """Scale box coordinates to a new canvas size."""
    h, w = boxes.canvas_size
    sx, sy = new_w / w, new_h / h
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = _cat([x1 * sx, y1 * sy, x2 * sx, y2 * sy], 1)
    return _rebuild(boxes, new, (new_h, new_w))


def crop_boxes(
    boxes: BoundingBoxes, top: int, left: int, height: int, width: int
) -> BoundingBoxes:
    """Translate boxes into a crop window and clip to its bounds."""
    xy = to_xyxy(boxes)
    x1 = lucid.clip(xy[:, 0:1] - left, 0.0, float(width))
    y1 = lucid.clip(xy[:, 1:2] - top, 0.0, float(height))
    x2 = lucid.clip(xy[:, 2:3] - left, 0.0, float(width))
    y2 = lucid.clip(xy[:, 3:4] - top, 0.0, float(height))
    new = _cat([x1, y1, x2, y2], 1)
    return _rebuild(boxes, new, (height, width))


def pad_boxes(
    boxes: BoundingBoxes, left: int, top: int, new_h: int, new_w: int
) -> BoundingBoxes:
    """Shift boxes by a pad offset onto a larger canvas."""
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = _cat([x1 + left, y1 + top, x2 + left, y2 + top], 1)
    return _rebuild(boxes, new, (new_h, new_w))


# ── keypoints ───────────────────────────────────────────────────────


def _kp_xy_rest(kps: Keypoints) -> tuple[Tensor, Tensor, Tensor | None]:
    """Split keypoint data into ``x``, ``y``, and any trailing columns."""
    d = kps.data
    x = d[:, 0:1]
    y = d[:, 1:2]
    rest = d[:, 2:] if int(d.shape[1]) > 2 else None
    return x, y, rest


def _kp_rebuild(
    x: Tensor, y: Tensor, rest: Tensor | None, canvas: tuple[int, int]
) -> Keypoints:
    cols = [x, y] if rest is None else [x, y, rest]
    return Keypoints(_cat(cols, 1), canvas)


def flip_keypoints(kps: Keypoints, *, horizontal: bool) -> Keypoints:
    """Mirror keypoints within the canvas."""
    h, w = kps.canvas_size
    x, y, rest = _kp_xy_rest(kps)
    if horizontal:
        x = w - x
    else:
        y = h - y
    return _kp_rebuild(x, y, rest, kps.canvas_size)


def resize_keypoints(kps: Keypoints, new_h: int, new_w: int) -> Keypoints:
    """Scale keypoint coordinates to a new canvas size."""
    h, w = kps.canvas_size
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(x * (new_w / w), y * (new_h / h), rest, (new_h, new_w))


def crop_keypoints(
    kps: Keypoints, top: int, left: int, height: int, width: int
) -> Keypoints:
    """Translate keypoints into a crop window (canvas → crop size)."""
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(x - left, y - top, rest, (height, width))


def pad_keypoints(
    kps: Keypoints, left: int, top: int, new_h: int, new_w: int
) -> Keypoints:
    """Shift keypoints by a pad offset onto a larger canvas."""
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(x + left, y + top, rest, (new_h, new_w))


# ── affine / transpose / rot90 (boxes + keypoints) ──────────────────


def _clip_xyxy(
    x1: Tensor, y1: Tensor, x2: Tensor, y2: Tensor, h: int, w: int
) -> Tensor:
    return _cat(
        [
            lucid.clip(x1, 0.0, float(w)),
            lucid.clip(y1, 0.0, float(h)),
            lucid.clip(x2, 0.0, float(w)),
            lucid.clip(y2, 0.0, float(h)),
        ],
        dim=1,
    )


def affine_boxes(
    boxes: "BoundingBoxes", matrix: Tensor, out_hw: tuple[int, int]
) -> "BoundingBoxes":
    """Warp boxes by a forward matrix; return the enclosing axis-aligned box."""
    from lucid.utils.transforms import functional as _F

    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    # Four corners stacked as (4N, 2).
    corners = _cat(
        [
            _cat([x1, y1], 1),
            _cat([x2, y1], 1),
            _cat([x2, y2], 1),
            _cat([x1, y2], 1),
        ],
        dim=0,
    )
    warped = _F.affine_points(corners, matrix)  # (4N, 2)
    n = int(xy.shape[0])
    wx = warped[:, 0:1].reshape(4, n)
    wy = warped[:, 1:2].reshape(4, n)
    nx1 = lucid.min(wx, dim=0, keepdim=True).reshape(n, 1)
    nx2 = lucid.max(wx, dim=0, keepdim=True).reshape(n, 1)
    ny1 = lucid.min(wy, dim=0, keepdim=True).reshape(n, 1)
    ny2 = lucid.max(wy, dim=0, keepdim=True).reshape(n, 1)
    new = _clip_xyxy(nx1, ny1, nx2, ny2, out_hw[0], out_hw[1])
    return _rebuild(boxes, new, out_hw)


def affine_keypoints(
    kps: "Keypoints", matrix: Tensor, out_hw: tuple[int, int]
) -> "Keypoints":
    """Warp keypoint coordinates by a forward matrix."""
    from lucid.utils.transforms import functional as _F

    x, y, rest = _kp_xy_rest(kps)
    xy = _F.affine_points(_cat([x, y], 1), matrix)
    return _kp_rebuild(xy[:, 0:1], xy[:, 1:2], rest, out_hw)


def transpose_boxes(boxes: "BoundingBoxes") -> "BoundingBoxes":
    """Swap x/y axes (image transpose); canvas (H, W) -> (W, H)."""
    h, w = boxes.canvas_size
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = _cat([y1, x1, y2, x2], 1)
    return _rebuild(boxes, new, (w, h))


def transpose_keypoints(kps: "Keypoints") -> "Keypoints":
    """Swap x/y of keypoints; canvas (H, W) -> (W, H)."""
    h, w = kps.canvas_size
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(y, x, rest, (w, h))


def rot90_boxes(boxes: "BoundingBoxes", k: int) -> "BoundingBoxes":
    """Rotate boxes by ``k`` CCW quarter-turns (matching ``rot90``)."""
    out = boxes
    for _ in range(k % 4):
        h, w = out.canvas_size
        xy = to_xyxy(out)
        x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
        # CCW: x' = y, y' = (W-1) - x; canvas -> (W, H).
        nx1, nx2 = y1, y2
        ny1 = (w - 1) - x2
        ny2 = (w - 1) - x1
        new = _cat([nx1, ny1, nx2, ny2], 1)
        out = _rebuild(out, new, (w, h))
    return out


def rot90_keypoints(kps: "Keypoints", k: int) -> "Keypoints":
    """Rotate keypoints by ``k`` CCW quarter-turns."""
    out = kps
    for _ in range(k % 4):
        h, w = out.canvas_size
        x, y, rest = _kp_xy_rest(out)
        out = _kp_rebuild(y, (w - 1) - x, rest, (w, h))
    return out
