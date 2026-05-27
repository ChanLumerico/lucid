"""Core transform hierarchy — generic, typed, multi-target.

Design
------
Every transform is generic over a **parameter type** ``P`` (a frozen
dataclass).  :meth:`Transform.make_params` samples ``P`` once per call
(from the sample's image); the typed ``_apply_*`` hooks receive it
without casts.  This is what keeps randomized transforms reproducible
*and* statically type-checked.

Class hierarchy::

    Transform[P]                 (ABC) make_params + _apply_{image,mask,boxes} + dispatch
    ├─ GeometricTransform[P]     (ABC) spatial — _apply_mask / _apply_boxes are *required*
    └─ PhotometricTransform[P]   (ABC) colour — mask/boxes left untouched (identity)

Mixins compose orthogonal behaviour:

    _NoParams       make_params → NO_PARAMS         (deterministic transforms)
    _ProbabilityGate  p + _gate()                   (Bernoulli-applied transforms)

A transform may be called on a single image tensor *or* a structured
**sample** mixing :class:`~lucid.utils.transforms.Image` /
:class:`~lucid.utils.transforms.Mask` /
:class:`~lucid.utils.transforms.BoundingBoxes` nested in dict / list /
tuple — every target is routed to its typed hook with the *same*
``params`` so an image, its mask, and its boxes move together.  A bare
:class:`lucid.Tensor` is treated as an image and returned as a tensor.
"""

import abc
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms._datatypes import BoundingBoxes, Image, Mask


@dataclass(frozen=True)
class Empty:
    """Parameter type for deterministic transforms (no per-call state)."""


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
    if isinstance(obj, Image):
        return obj.data
    if isinstance(obj, Mask):
        return obj.data
    if isinstance(obj, BoundingBoxes):
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

    Subclasses implement :meth:`make_params` (the per-call parameter
    draw) and :meth:`_apply_image`.  Mask / box handling defaults to
    identity here; :class:`GeometricTransform` makes them required.
    """

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

    def _dispatch(self, obj: object, params: P) -> object:
        if isinstance(obj, Image):
            return Image(self._apply_image(obj.data, params))
        if isinstance(obj, Mask):
            return Mask(self._apply_mask(obj.data, params))
        if isinstance(obj, BoundingBoxes):
            return self._apply_boxes(obj, params)
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
        return self._dispatch(inputs, self.make_params(ref))


class GeometricTransform(Transform[P], abc.ABC):
    """Spatial transform — *must* move masks and boxes with the image.

    Re-declaring the mask / box hooks as abstract turns "a geometric
    transform handles all three target types" into a contract the type
    checker enforces, preventing the silent bug where a new geometric
    transform forgets to resample its mask or shift its boxes.
    """

    @abc.abstractmethod
    def _apply_mask(self, mask: Tensor, params: P) -> Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def _apply_boxes(self, boxes: BoundingBoxes, params: P) -> BoundingBoxes:
        raise NotImplementedError


class PhotometricTransform(Transform[P], abc.ABC):
    """Colour / intensity transform — masks and boxes pass through.

    Inherits the identity ``_apply_mask`` / ``_apply_boxes`` from
    :class:`Transform`; the class exists to state intent and to host
    shared photometric helpers.
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


class _ProbabilityGate:
    """Mixin: Bernoulli gate (apply with probability ``p``)."""

    def __init__(self, p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"probability p must be in [0, 1], got {p}")
        self.p = p

    def _gate(self) -> bool:
        return _random.rand() < self.p


class Compose(_NoParams, Transform[Empty]):
    """Chain transforms into one callable (works on tensors or samples).

    Parameters
    ----------
    transforms : list of Transform
        Applied left-to-right; each transform's output feeds the next.

    Examples
    --------
    >>> from lucid.utils.transforms import Compose, Resize, CenterCrop, Normalize
    >>> tf = Compose([
    ...     Resize(256),
    ...     CenterCrop(224),
    ...     Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ... ])
    >>> y = tf(image)
    """

    def __init__(self, transforms: list[TransformLike]) -> None:
        self.transforms = list(transforms)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        out: object = img
        for tf in self.transforms:
            out = tf(out)
        return out  # type: ignore[return-value]

    def __call__(self, inputs: object) -> object:
        for tf in self.transforms:
            inputs = tf(inputs)
        return inputs

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"
