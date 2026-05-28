"""Core transform hierarchy — generic, typed, multi-target.

Design
------
Every transform is generic over a **parameter type** ``P`` (a frozen
dataclass).  :meth:`Transform.make_params` samples ``P`` once per call
(from the sample's image); the typed ``_apply_*`` hooks receive it
without casts.  Every transform also carries a probability ``p``: the
*whole* transform is applied with probability ``p`` and is the identity
otherwise (Albumentations semantics).

Class hierarchy::

    Transform[P]                 (ABC) p-gate + make_params + _apply_* + dispatch
    ├─ GeometricTransform[P]     (ABC) spatial — mask / boxes / keypoints required
    └─ PhotometricTransform[P]   (ABC) colour — non-image targets pass through

Mixin: ``_NoParams`` (deterministic transforms → ``NO_PARAMS``).

A transform may be called on a single image tensor *or* a structured
**sample** mixing :class:`~lucid.utils.transforms.Image` /
:class:`~lucid.utils.transforms.Mask` /
:class:`~lucid.utils.transforms.BoundingBoxes` /
:class:`~lucid.utils.transforms.Keypoints` nested in dict / list /
tuple — every target is routed to its typed hook with the *same*
``params``.  A bare :class:`lucid.Tensor` is treated as an image.
"""

import abc
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    Image,
    Keypoints,
    Mask,
)


@dataclass(frozen=True)
class Empty:
    r"""Parameter type for deterministic transforms (no per-call state).

    Used by transforms whose behaviour is fully determined at
    construction time — :class:`Crop`, :class:`Normalize`,
    :class:`Compose`, etc.  The single shared instance
    :data:`NO_PARAMS` is returned by every such transform's
    :meth:`make_params` so the dispatch hooks can stay generic over
    ``P`` without allocating per call.
    """


NO_PARAMS = Empty()

P = TypeVar("P")


@runtime_checkable
class TransformLike(Protocol):
    """Structural type for anything callable on a sample.

    Every :class:`Transform` satisfies it regardless of its parameter
    type, so containers like :class:`Compose` can hold a heterogeneous
    list of transforms without fighting ``Transform[P]``'s invariance.
    """

    def __call__(self, inputs: object) -> object: ...


def _find_reference(obj: object) -> Tensor | None:
    """Find the image-like tensor in a sample, to size random params."""
    if isinstance(obj, (Image, Mask)):
        return obj.data
    if isinstance(obj, (BoundingBoxes, Keypoints)):
        return None
    if isinstance(obj, dict):
        for value in obj.values():
            ref = _find_reference(value)
            if ref is not None:
                return ref
        return None
    if isinstance(obj, (list, tuple)):
        for value in obj:
            ref = _find_reference(value)
            if ref is not None:
                return ref
        return None
    if isinstance(obj, Tensor):
        return obj
    return None


class Transform(Generic[P], abc.ABC):
    """Abstract base for a transform parameterized by its sample-params ``P``.

    Parameters
    ----------
    p : float, optional, default=1.0
        Probability of applying the transform; otherwise the input
        passes through unchanged.
    """

    def __init__(self, p: float = 1.0) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability p must be in [0, 1], got {p}")
        self.p = p

    @abc.abstractmethod
    def make_params(self, img: Tensor) -> P:
        """Sample the per-call parameters from the reference image."""
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_image(self, img: Tensor, params: P) -> Tensor:
        """Apply the transform to an image tensor."""
        raise NotImplementedError

    def _apply_mask(self, mask: Tensor, params: P) -> Tensor:
        """Apply to a mask (default identity; overridden by geometric)."""
        return mask

    def _apply_boxes(self, boxes: BoundingBoxes, params: P) -> BoundingBoxes:
        """Apply to boxes (default identity; overridden by geometric)."""
        return boxes

    def _apply_keypoints(self, kps: Keypoints, params: P) -> Keypoints:
        """Apply to keypoints (default identity; overridden by geometric)."""
        return kps

    def _dispatch(self, obj: object, params: P) -> object:
        if isinstance(obj, Image):
            return Image(self._apply_image(obj.data, params))
        if isinstance(obj, Mask):
            return Mask(self._apply_mask(obj.data, params))
        if isinstance(obj, BoundingBoxes):
            return self._apply_boxes(obj, params)
        if isinstance(obj, Keypoints):
            return self._apply_keypoints(obj, params)
        if isinstance(obj, dict):
            return {key: self._dispatch(val, params) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._dispatch(val, params) for val in obj)
        if isinstance(obj, Tensor):
            return self._apply_image(obj, params)
        return obj

    def __call__(self, inputs: object) -> object:
        """Transform a tensor, a typed target, or a nested sample."""
        ref = _find_reference(inputs)
        if ref is None:
            raise ValueError(
                f"{type(self).__name__}: no image / mask in the sample to derive "
                "transform parameters from"
            )
        if self.p < 1.0 and _random.rand() >= self.p:
            return inputs
        return self._dispatch(inputs, self.make_params(ref))


class GeometricTransform(Transform[P], abc.ABC):
    """Spatial transform — *must* move masks, boxes, and keypoints.

    Re-declaring the companion hooks as abstract turns "a geometric
    transform handles every target type" into a contract mypy enforces,
    preventing the silent bug where a new geometric transform forgets to
    move one of them.
    """

    @abc.abstractmethod
    def _apply_mask(self, mask: Tensor, params: P) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_boxes(self, boxes: BoundingBoxes, params: P) -> BoundingBoxes:
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_keypoints(self, kps: Keypoints, params: P) -> Keypoints:
        raise NotImplementedError


class PhotometricTransform(Transform[P], abc.ABC):
    """Colour / intensity transform — non-image targets pass through.

    Inherits the identity companion hooks from :class:`Transform`; the
    class states intent and hosts shared photometric helpers.
    """

    @staticmethod
    def _require_channels(img: Tensor, expected: int) -> None:
        c = int(img.shape[-3])
        if c != expected:
            raise ValueError(f"expected {expected}-channel image, got {c} channels")


class _NoParams:
    """Mixin: deterministic transforms have no per-call parameters."""

    def make_params(self, img: Tensor) -> Empty:
        return NO_PARAMS


@dataclass
class BboxParams:
    r"""Bounding-box handling policy for :class:`Compose` (Albumentations-style).

    When a :class:`Compose` is constructed with ``bbox_params=...``,
    it records every :class:`~lucid.utils.transforms.BoundingBoxes`'s
    *pre-pipeline* area, runs the pipeline, then calls
    :func:`~lucid.utils.transforms._datatypes.filter_boxes` on the
    result with these thresholds.  Out-of-frame / heavily-occluded
    boxes — and their labels — are dropped in one place rather than
    re-implemented inside every geometric transform.

    Parameters
    ----------
    format : str, optional, default="pascal_voc"
        Albumentations format name.  Informational only; each
        :class:`~lucid.utils.transforms.BoundingBoxes` already carries
        its own ``format`` and conversion is handled per-instance.
    min_area : float, optional, default=0.0
        Drop post-transform boxes whose absolute pixel area is below
        this (units: pixels² in the post-transform canvas).
    min_visibility : float, optional, default=0.0
        Drop boxes whose visible fraction (area / original area) falls
        below this.  The ratio is computed against the
        :func:`~lucid.utils.transforms._datatypes.box_areas` snapshot
        recorded *before* the pipeline ran.
    label_fields : tuple of str, optional
        Reserved for API parity with Albumentations; in Lucid, labels
        already travel on :attr:`BoundingBoxes.labels` and are trimmed
        in lock-step.  No-op here.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.Compose(
    ...     [T.Crop(0, 0, 50, 50)],
    ...     bbox_params=T.BboxParams(min_area=10.0, min_visibility=0.5),
    ... )
    """

    format: str = "pascal_voc"
    min_area: float = 0.0
    min_visibility: float = 0.0
    label_fields: tuple[str, ...] = ()


def _iter_boxes(obj: object) -> list[BoundingBoxes]:
    """Collect every :class:`BoundingBoxes` in a sample (traversal order)."""
    found: list[BoundingBoxes] = []

    def _walk(o: object) -> None:
        if isinstance(o, BoundingBoxes):
            found.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                _walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                _walk(v)

    _walk(obj)
    return found


def _filter_boxes_in_sample(
    obj: object, orig_areas: list[list[float]], params: BboxParams, counter: list[int]
) -> object:
    """Replace each :class:`BoundingBoxes` with its filtered version."""
    from lucid.utils.transforms._datatypes import filter_boxes

    if isinstance(obj, BoundingBoxes):
        idx = counter[0]
        counter[0] += 1
        areas = orig_areas[idx] if idx < len(orig_areas) else None
        return filter_boxes(
            obj,
            orig_areas=areas,
            min_area=params.min_area,
            min_visibility=params.min_visibility,
        )
    if isinstance(obj, dict):
        return {
            k: _filter_boxes_in_sample(v, orig_areas, params, counter)
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        return type(obj)(
            _filter_boxes_in_sample(v, orig_areas, params, counter) for v in obj
        )
    return obj


class Compose(_NoParams, Transform[Empty]):
    """Chain transforms into one callable (works on tensors or samples).

    Parameters
    ----------
    transforms : list of Transform
        Applied left-to-right; each transform's output feeds the next.
    bbox_params : BboxParams, optional
        When given, bounding boxes are filtered after the pipeline by
        ``min_area`` / ``min_visibility`` (degenerate boxes that left the
        frame are dropped, along with their labels).

    Examples
    --------
    >>> from lucid.utils.transforms import Compose, Resize, CenterCrop, Normalize
    >>> tf = Compose([
    ...     Resize(256, 256),
    ...     CenterCrop(224, 224),
    ...     Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
    ...               max_pixel_value=1.0),
    ... ])
    >>> y = tf(image)
    """

    def __init__(
        self, transforms: list[TransformLike], bbox_params: BboxParams | None = None
    ) -> None:
        super().__init__(p=1.0)
        self.transforms = list(transforms)
        self.bbox_params = bbox_params

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        out: object = img
        for tf in self.transforms:
            out = tf(out)
        return out  # type: ignore[return-value]

    def __call__(self, inputs: object) -> object:
        from lucid.utils.transforms._datatypes import box_areas

        orig_areas: list[list[float]] = []
        if self.bbox_params is not None:
            orig_areas = [box_areas(b) for b in _iter_boxes(inputs)]
        for tf in self.transforms:
            inputs = tf(inputs)
        if self.bbox_params is not None:
            inputs = _filter_boxes_in_sample(inputs, orig_areas, self.bbox_params, [0])
        return inputs

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"
