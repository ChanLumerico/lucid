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
#   xyxy / xywh / cxcywh — absolute pixel coordinates.
#   albumentations       — normalized xyxy (x1,y1,x2,y2 in [0, 1]).
#   yolo                 — normalized cxcywh (cx,cy,w,h in [0, 1]).
# Albumentations names map: pascal_voc→xyxy, coco→xywh.
_BOX_FORMATS = frozenset({"xyxy", "xywh", "cxcywh", "albumentations", "yolo"})

# Albumentations format-name aliases → internal names.
_ALBU_FORMAT_ALIASES = {"pascal_voc": "xyxy", "coco": "xywh"}


def normalize_box_format(fmt: str) -> str:
    r"""Map an Albumentations format alias to the canonical internal name.

    Albumentations exposes two human-friendly aliases (``"pascal_voc"``
    for the corner-corner format and ``"coco"`` for the corner-size
    format); the internal representation uses the shorter
    ``"xyxy"`` / ``"xywh"`` names.  Unknown values pass through
    untouched so :class:`BoundingBoxes` can raise a clear error.

    Parameters
    ----------
    fmt : str
        A user-supplied format name.  Either an internal name
        (``"xyxy"``, ``"xywh"``, ``"cxcywh"``, ``"yolo"``,
        ``"albumentations"``) or an Albumentations alias
        (``"pascal_voc"``, ``"coco"``).

    Returns
    -------
    str
        The canonical internal name (one of ``"xyxy"``, ``"xywh"``,
        ``"cxcywh"``, ``"yolo"``, ``"albumentations"``), or ``fmt``
        unchanged if it isn't a known alias.

    Examples
    --------
    >>> normalize_box_format("pascal_voc")
    'xyxy'
    >>> normalize_box_format("coco")
    'xywh'
    >>> normalize_box_format("yolo")
    'yolo'
    """
    return _ALBU_FORMAT_ALIASES.get(fmt, fmt)


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
    r"""Wraps ``(N, 4)`` bounding boxes with a format + canvas size.

    A typed target consumed by every :class:`GeometricTransform` —
    flips / crops / resizes update both the coordinates and the
    reported ``canvas_size`` consistently.  Optional ``labels`` ride
    along the same row dimension so :func:`filter_boxes` can drop
    out-of-frame boxes + their labels in one step.

    Parameters
    ----------
    data : Tensor
        ``(N, 4)`` coordinates interpreted under ``format``.  For the
        absolute formats (``xyxy`` / ``xywh`` / ``cxcywh``) units are
        pixels; for the normalized formats (``yolo`` / ``albumentations``)
        units are fractions of ``canvas_size``.
    format : {"xyxy", "xywh", "cxcywh", "yolo", "albumentations", \
              "pascal_voc", "coco"}
        Coordinate convention.  ``"pascal_voc"`` is an alias for
        ``"xyxy"``; ``"coco"`` for ``"xywh"``.  See
        :func:`normalize_box_format`.
    canvas_size : (int, int)
        ``(H, W)`` of the image the boxes index into.  Geometric
        transforms (crop / resize / pad / rotate) re-emit a new
        :class:`BoundingBoxes` with the matching canvas.
    labels : Tensor, optional
        ``(N,)`` class indices (or any per-box scalar — score, track
        id, ...).  Carried through every geometric transform; trimmed
        in lock-step with rows by :func:`filter_boxes`.  Default
        ``None`` (no per-box labels).

    Raises
    ------
    ValueError
        If ``format`` (after alias normalisation) is not one of the
        recognised names.

    Notes
    -----
    Format conventions (each row's 4 values):

    * ``xyxy`` — ``(x_min, y_min, x_max, y_max)``
    * ``xywh`` — ``(x_min, y_min, width, height)``
    * ``cxcywh`` — ``(cx, cy, width, height)``
    * ``yolo`` — ``(cx, cy, w, h)``, all in ``[0, 1]``
    * ``albumentations`` — ``(x_min, y_min, x_max, y_max)``, all in ``[0, 1]``

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> boxes = T.BoundingBoxes(
    ...     lucid.tensor([[10.0, 10.0, 40.0, 40.0]]),
    ...     "xyxy",
    ...     canvas_size=(100, 100),
    ...     labels=lucid.tensor([1.0]),
    ... )
    >>> out = T.HorizontalFlip(p=1.0)({"image": T.Image(lucid.rand(3, 100, 100)),
    ...                                "boxes": boxes})
    >>> out["boxes"].labels.numpy().tolist()
    [1.0]
    """

    data: Tensor
    format: str
    canvas_size: tuple[int, int]
    labels: Tensor | None = None

    def __post_init__(self) -> None:
        self.format = normalize_box_format(self.format)
        if self.format not in _BOX_FORMATS:
            raise ValueError(
                f"BoundingBoxes: unknown format {self.format!r}; "
                f"expected one of {sorted(_BOX_FORMATS)} (or pascal_voc/coco)"
            )


def to_xyxy(boxes: BoundingBoxes) -> Tensor:
    r"""Return the box coordinates as ``(N, 4)`` absolute-pixel xyxy.

    Normalisation-free output: regardless of whether ``boxes`` is in
    ``yolo`` (normalized cxcywh) or ``albumentations`` (normalized
    xyxy), the returned tensor is in pixel units of
    ``boxes.canvas_size``.  This is the canonical pivot every
    transform uses internally — flips / crops / resizes operate on
    xyxy and :func:`from_xyxy` re-encodes back to the original format.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes in any supported format.

    Returns
    -------
    Tensor
        ``(N, 4)`` absolute-pixel ``xyxy``.  Identical to ``boxes.data``
        when the source format is already ``xyxy``.
    """
    d = boxes.data
    h, w = boxes.canvas_size
    fmt = boxes.format
    if fmt == "xyxy":
        return d
    a, b, c, e = d[:, 0:1], d[:, 1:2], d[:, 2:3], d[:, 3:4]
    if fmt == "xywh":
        return _cat([a, b, a + c, b + e], 1)
    if fmt == "cxcywh":
        return _cat([a - c / 2, b - e / 2, a + c / 2, b + e / 2], 1)
    if fmt == "albumentations":  # normalized xyxy
        return _cat([a * w, b * h, c * w, e * h], 1)
    # yolo: normalized cxcywh
    cx, cy, bw, bh = a * w, b * h, c * w, e * h
    return _cat([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)


def from_xyxy(xyxy: Tensor, fmt: str, canvas: tuple[int, int]) -> Tensor:
    r"""Convert absolute ``(N, 4)`` xyxy coordinates back into ``fmt``.

    Inverse of :func:`to_xyxy`.  Normalised output formats divide by
    ``canvas`` so the round-trip ``from_xyxy(to_xyxy(b), b.format,
    b.canvas_size)`` reproduces the original ``data`` tensor.

    Parameters
    ----------
    xyxy : Tensor
        ``(N, 4)`` absolute-pixel ``xyxy`` coordinates.
    fmt : str
        Target format (any of ``xyxy``, ``xywh``, ``cxcywh``,
        ``albumentations``, ``yolo``).  Pass the alias-normalised name
        — use :func:`normalize_box_format` first if you have an Albu
        alias.
    canvas : (int, int)
        ``(H, W)`` used to normalise when ``fmt`` is ``"yolo"`` or
        ``"albumentations"``; ignored for the absolute formats.

    Returns
    -------
    Tensor
        ``(N, 4)`` coordinates in ``fmt``.
    """
    h, w = canvas
    x1, y1, x2, y2 = xyxy[:, 0:1], xyxy[:, 1:2], xyxy[:, 2:3], xyxy[:, 3:4]
    if fmt == "xyxy":
        return xyxy
    if fmt == "xywh":
        return _cat([x1, y1, x2 - x1, y2 - y1], 1)
    if fmt == "cxcywh":
        return _cat([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], 1)
    if fmt == "albumentations":
        return _cat([x1 / w, y1 / h, x2 / w, y2 / h], 1)
    # yolo
    return _cat([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h], 1)


def _rebuild(
    boxes: BoundingBoxes, xyxy: Tensor, canvas: tuple[int, int]
) -> BoundingBoxes:
    """Re-wrap transformed xyxy coords back into the original format + labels."""
    return BoundingBoxes(
        from_xyxy(xyxy, boxes.format, canvas), boxes.format, canvas, boxes.labels
    )


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
    boxes: BoundingBoxes, matrix: Tensor, out_hw: tuple[int, int]
) -> BoundingBoxes:
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
    kps: Keypoints, matrix: Tensor, out_hw: tuple[int, int]
) -> Keypoints:
    """Warp keypoint coordinates by a forward matrix."""
    from lucid.utils.transforms import functional as _F

    x, y, rest = _kp_xy_rest(kps)
    xy = _F.affine_points(_cat([x, y], 1), matrix)
    return _kp_rebuild(xy[:, 0:1], xy[:, 1:2], rest, out_hw)


def transpose_boxes(boxes: BoundingBoxes) -> BoundingBoxes:
    """Swap x/y axes (image transpose); canvas (H, W) -> (W, H)."""
    h, w = boxes.canvas_size
    xy = to_xyxy(boxes)
    x1, y1, x2, y2 = xy[:, 0:1], xy[:, 1:2], xy[:, 2:3], xy[:, 3:4]
    new = _cat([y1, x1, y2, x2], 1)
    return _rebuild(boxes, new, (w, h))


def transpose_keypoints(kps: Keypoints) -> Keypoints:
    """Swap x/y of keypoints; canvas (H, W) -> (W, H)."""
    h, w = kps.canvas_size
    x, y, rest = _kp_xy_rest(kps)
    return _kp_rebuild(y, x, rest, (w, h))


def rot90_boxes(boxes: BoundingBoxes, k: int) -> BoundingBoxes:
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


def rot90_keypoints(kps: Keypoints, k: int) -> Keypoints:
    """Rotate keypoints by ``k`` CCW quarter-turns."""
    out = kps
    for _ in range(k % 4):
        h, w = out.canvas_size
        x, y, rest = _kp_xy_rest(out)
        out = _kp_rebuild(y, (w - 1) - x, rest, (w, h))
    return out


# ── bounding-box area + filtering (detection pipelines) ─────────────


def box_areas(boxes: BoundingBoxes) -> list[float]:
    r"""Absolute pixel area of every box, as a Python ``list[float]``.

    Computes :math:`\max(0, (x_2 - x_1)(y_2 - y_1))` on the
    :func:`to_xyxy` representation.  Used by :func:`filter_boxes` to
    drop degenerate / too-small / too-occluded rows, and by
    :class:`~lucid.utils.transforms.Compose` to record pre-pipeline
    areas for the ``min_visibility`` calculation.

    Parameters
    ----------
    boxes : BoundingBoxes
        Source boxes in any supported format.

    Returns
    -------
    list of float
        ``N`` entries, one per box, clamped at 0.  A Python list (not
        a Tensor) so callers can stash it across the pipeline without
        carrying an autograd handle.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> b = T.BoundingBoxes(lucid.tensor([[0.0, 0.0, 10.0, 5.0]]),
    ...                     "xyxy", (100, 100))
    >>> box_areas(b)
    [50.0]
    """
    xy = to_xyxy(boxes)
    n = int(xy.shape[0])
    w = xy[:, 2] - xy[:, 0]
    h = xy[:, 3] - xy[:, 1]
    area = w * h
    return [max(float(area[i].item()), 0.0) for i in range(n)]


def filter_boxes(
    boxes: BoundingBoxes,
    *,
    orig_areas: list[float] | None = None,
    min_area: float = 0.0,
    min_visibility: float = 0.0,
) -> BoundingBoxes:
    r"""Drop degenerate / too-small / too-occluded boxes (+ their labels).

    A box at row ``i`` is dropped when **any** of the following holds:

    * its post-transform area :math:`\le 0` (degenerate),
    * its post-transform area :math:`<` ``min_area`` (absolute pixels),
    * ``orig_areas`` is supplied **and** the visible fraction
      :math:`a_i / o_i <` ``min_visibility`` (where :math:`a_i, o_i`
      are the post-transform and original areas).

    The companion ``labels`` tensor — when present — is trimmed in
    lock-step, so downstream loss / NMS code never sees a label
    without a matching box.

    Parameters
    ----------
    boxes : BoundingBoxes
        Boxes (and optional labels) after the transform whose effect
        you want to filter.
    orig_areas : list of float, optional
        Per-row areas captured *before* the transform was applied —
        typically the output of :func:`box_areas` on the pre-pipeline
        boxes, recorded by :class:`~lucid.utils.transforms.Compose`
        with ``bbox_params``.  When omitted, ``min_visibility`` is
        ignored (only ``min_area`` / degeneracy are checked).
    min_area : float, optional, default=0.0
        Absolute-pixel minimum area; rows below this are dropped.
    min_visibility : float, optional, default=0.0
        Minimum fraction of the original area that must still be
        visible.  Requires ``orig_areas``.

    Returns
    -------
    BoundingBoxes
        A new instance carrying only the surviving rows + their
        labels.  Format, canvas, and the labels-or-``None`` shape are
        all preserved.

    Examples
    --------
    Filter out a tiny box and a degenerate one (with labels):

    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> b = T.BoundingBoxes(
    ...     lucid.tensor([[0., 0., 30., 30.],   # area 900
    ...                   [0., 0.,  0.,  0.],   # degenerate
    ...                   [0., 0.,  2.,  2.]]), # area 4 (too small)
    ...     "xyxy", (100, 100),
    ...     labels=lucid.tensor([1.0, 2.0, 3.0]),
    ... )
    >>> out = filter_boxes(b, min_area=10.0)
    >>> out.labels.numpy().tolist()
    [1.0]
    """
    cur = box_areas(boxes)
    keep: list[int] = []
    for i, area in enumerate(cur):
        if area <= 0.0 or area < min_area:
            continue
        if (
            orig_areas is not None
            and i < len(orig_areas)
            and orig_areas[i] > 0.0
            and (area / orig_areas[i]) < min_visibility
        ):
            continue
        keep.append(i)

    if not keep:
        empty_lbl = boxes.labels[:0] if boxes.labels is not None else None
        return BoundingBoxes(boxes.data[:0], boxes.format, boxes.canvas_size, empty_lbl)

    data = _cat([boxes.data[i : i + 1] for i in keep], 0)
    labels = (
        _cat([boxes.labels[i : i + 1] for i in keep], 0)
        if boxes.labels is not None
        else None
    )
    return BoundingBoxes(data, boxes.format, boxes.canvas_size, labels)
