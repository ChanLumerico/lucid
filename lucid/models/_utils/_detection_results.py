"""Detection results — the per-image dict every detector's postprocess returns.

Kept apart from ``_detection`` (box ops, NMS, anchors, RoI ops) because
it is the *output contract* rather than an algorithm: one place decides the
keys and dtypes, so the detectors cannot drift apart again.  They did —
several returned float32 class ids built with ``lucid.full(..., float(c))``.
"""

import lucid
from lucid._tensor.tensor import Tensor
from lucid._types import DeviceLike


def detection_result(
    boxes: Tensor,
    scores: Tensor,
    labels: Tensor,
    masks: Tensor | None = None,
) -> dict[str, Tensor]:
    """One image's postprocess result — the dict every detector returns.

    Parameters
    ----------
    boxes : Tensor
        ``(K, 4)`` xyxy pixel boxes.
    scores : Tensor
        ``(K,)`` detection scores.
    labels : Tensor
        ``(K,)`` class ids, in whatever dtype the caller assembled them.
    masks : Tensor or None, optional
        ``(K, 1, H, W)`` instance masks; the key is present only when given.

    Returns
    -------
    dict[str, Tensor]
        ``"boxes"``, ``"scores"``, ``"labels"`` (and ``"masks"``).
        ``labels`` is always int64 — the dtype detection *targets* carry —
        so a result can be compared with, or fed back as, a target without
        a cast.  Several detectors built theirs with ``lucid.full(...,
        float(c))`` and returned float32 class ids.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection_results import detection_result
    >>> out = detection_result(
    ...     lucid.zeros((1, 4)), lucid.ones((1,)), lucid.tensor([3.0])
    ... )
    >>> out["labels"].dtype
    lucid.int64
    """
    out = {"boxes": boxes, "scores": scores, "labels": labels.long()}
    if masks is not None:
        out["masks"] = masks
    return out


def empty_detection_result(
    device: DeviceLike, mask_hw: tuple[int, int] | None = None
) -> dict[str, Tensor]:
    """An image with no detections, each field typed as a non-empty result.

    Parameters
    ----------
    device : DeviceLike
        Device of the model's predictions — an empty result built on the
        CPU made later concatenation with a Metal batch raise.
    mask_hw : tuple[int, int] or None, optional
        ``(H, W)`` of the instance masks, for detectors that return them;
        ``None`` omits the ``"masks"`` key.

    Returns
    -------
    dict[str, Tensor]
        ``(0, 4)`` boxes, ``(0,)`` scores, ``(0,)`` int64 labels (and
        ``(0, 1, H, W)`` masks).

    Examples
    --------
    >>> from lucid.models._utils._detection_results import empty_detection_result
    >>> out = empty_detection_result("cpu")
    >>> out["boxes"].shape, out["labels"].dtype
    ((0, 4), lucid.int64)
    """
    zeros = lucid.zeros((0,), device=device)
    masks = None
    if mask_hw is not None:
        masks = lucid.zeros((0, 1, mask_hw[0], mask_hw[1]), device=device)
    return detection_result(lucid.zeros((0, 4), device=device), zeros, zeros, masks)


def merge_class_detections(
    boxes: list[Tensor],
    scores: list[Tensor],
    labels: list[Tensor],
    device: DeviceLike,
    max_detections: int,
) -> dict[str, Tensor]:
    """Merge one image's per-class NMS survivors, best score first.

    Parameters
    ----------
    boxes, scores, labels : list[Tensor]
        One entry per class that kept anything: ``(k, 4)`` boxes, ``(k,)``
        scores and ``(k,)`` class ids.
    device : DeviceLike
        Device for the empty result when no class kept anything.
    max_detections : int
        Per-*image* cap, applied once after the global score sort.  Capping
        inside the class loop let each of the K classes keep a full quota,
        so the returned count scaled with the class count.

    Returns
    -------
    dict[str, Tensor]
        As :func:`detection_result`.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models._utils._detection_results import merge_class_detections
    >>> out = merge_class_detections(
    ...     [lucid.zeros((1, 4)), lucid.ones((1, 4))],
    ...     [lucid.tensor([0.2]), lucid.tensor([0.9])],
    ...     [lucid.tensor([0]), lucid.tensor([5])],
    ...     "cpu",
    ...     max_detections=10,
    ... )
    >>> out["labels"].tolist()
    [5, 0]
    """
    if not boxes:
        return empty_detection_result(device)
    all_b = lucid.cat(boxes, dim=0)
    all_s = lucid.cat(scores, dim=0)
    all_l = lucid.cat(labels, dim=0)
    order = lucid.argsort(-all_s)[:max_detections]
    return detection_result(all_b[order], all_s[order], all_l[order])
