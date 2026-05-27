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
        return lucid.concat([a, b, a + c, b + e], dim=1)  # type: ignore[arg-type]
    # cxcywh: (cx, cy, w, h) -> corners
    return lucid.concat(
        [a - c / 2, b - e / 2, a + c / 2, b + e / 2], dim=1  # type: ignore[arg-type]
    )


def from_xyxy(xyxy: Tensor, fmt: str) -> Tensor:
    """Convert ``(N, 4)`` xyxy coordinates back to ``fmt``."""
    if fmt == "xyxy":
        return xyxy
    x1 = xyxy[:, 0:1]
    y1 = xyxy[:, 1:2]
    x2 = xyxy[:, 2:3]
    y2 = xyxy[:, 3:4]
    if fmt == "xywh":
        return lucid.concat([x1, y1, x2 - x1, y2 - y1], dim=1)  # type: ignore[arg-type]
    # cxcywh
    return lucid.concat(
        [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=1  # type: ignore[arg-type]
    )


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
        new = lucid.concat([w - x2, y1, w - x1, y2], dim=1)  # type: ignore[arg-type]
    else:
        new = lucid.concat([x1, h - y2, x2, h - y1], dim=1)  # type: ignore[arg-type]
    return _rebuild(boxes, new, boxes.canvas_size)


def resize_boxes(boxes: BoundingBoxes, new_h: int, new_w: int) -> BoundingBoxes:
    """Scale box coordinates to a new canvas size."""
    h, w = boxes.canvas_size
    sx, sy = new_w / w, new_h / h
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = lucid.concat([x1 * sx, y1 * sy, x2 * sx, y2 * sy], dim=1)  # type: ignore[arg-type]
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
    new = lucid.concat([x1, y1, x2, y2], dim=1)  # type: ignore[arg-type]
    return _rebuild(boxes, new, (height, width))


def pad_boxes(
    boxes: BoundingBoxes, left: int, top: int, new_h: int, new_w: int
) -> BoundingBoxes:
    """Shift boxes by a pad offset onto a larger canvas."""
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = lucid.concat([x1 + left, y1 + top, x2 + left, y2 + top], dim=1)  # type: ignore[arg-type]
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
    return Keypoints(lucid.concat(cols, dim=1), canvas)  # type: ignore[arg-type]


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
