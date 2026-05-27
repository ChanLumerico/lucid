"""Core transform primitives — :class:`Transform` ABC + :class:`Compose`.

Lucid's transform library follows the torchvision-v2 design adapted to
the Lucid-tensor-only world (no numpy / PIL inside ``lucid/``, per H4):
transforms consume and produce :class:`lucid.Tensor` images shaped
``(C, H, W)`` or ``(B, C, H, W)``.  Image *decoding* (file → tensor)
lives outside this package.

Multi-target dispatch
---------------------
A transform may be called on a single image tensor *or* on a structured
**sample** mixing typed targets — :class:`~lucid.utils.transforms.Image`,
:class:`~lucid.utils.transforms.Mask`,
:class:`~lucid.utils.transforms.BoundingBoxes` — nested in dicts / lists
/ tuples.  Randomness is sampled **once** per call (from the sample's
image) in :meth:`make_params`, then every target is routed to its typed
hook with those same params so an image, its mask, and its boxes move
together:

* :meth:`_apply_image` — required.
* :meth:`_apply_mask`   — defaults to identity (photometric transforms
  leave masks alone); geometric transforms override.
* :meth:`_apply_boxes`  — defaults to identity; geometric transforms
  override.

A bare :class:`lucid.Tensor` is treated as an image and returned as a
tensor, so single-image pipelines are unaffected.
"""

import abc

from lucid._tensor import Tensor
from lucid.utils.transforms._datatypes import BoundingBoxes, Image, Mask


def _find_reference(obj: object) -> Tensor | None:
    """Find the image-like tensor in a sample, to size random params."""
    if isinstance(obj, Image):
        return obj.data
    if isinstance(obj, Mask):
        return obj.data
    if isinstance(obj, BoundingBoxes):
        return None
    if isinstance(obj, dict):
        for v in obj.values():
            ref = _find_reference(v)
            if ref is not None:
                return ref
        return None
    if isinstance(obj, (list, tuple)):
        for v in obj:
            ref = _find_reference(v)
            if ref is not None:
                return ref
        return None
    if isinstance(obj, Tensor):
        return obj
    return None


class Transform(abc.ABC):
    """Abstract base for a single transform.

    Subclasses implement :meth:`_apply_image` (required), and override
    :meth:`make_params` (randomness) / :meth:`_apply_mask` /
    :meth:`_apply_boxes` as needed.  Calling the transform routes a
    sample through all hooks with one shared parameter draw.
    """

    def make_params(self, img: Tensor) -> dict[str, object]:
        """Sample per-call parameters (randomness) from the image.

        Empty for deterministic transforms.  Randomized transforms draw
        their parameters here so the same draw applies across an image's
        mask / box companions.
        """
        return {}

    @abc.abstractmethod
    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        """Apply the transform to an image tensor given ``params``."""
        raise NotImplementedError

    def _apply_mask(self, mask: Tensor, params: dict[str, object]) -> Tensor:
        """Apply to a mask tensor.  Default: identity (override in geometric)."""
        return mask

    def _apply_boxes(
        self, boxes: BoundingBoxes, params: dict[str, object]
    ) -> BoundingBoxes:
        """Apply to bounding boxes.  Default: identity (override in geometric)."""
        return boxes

    def _dispatch(self, obj: object, params: dict[str, object]) -> object:
        if isinstance(obj, Image):
            return Image(self._apply_image(obj.data, params))
        if isinstance(obj, Mask):
            return Mask(self._apply_mask(obj.data, params))
        if isinstance(obj, BoundingBoxes):
            return self._apply_boxes(obj, params)
        if isinstance(obj, dict):
            return {k: self._dispatch(v, params) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._dispatch(v, params) for v in obj)
        if isinstance(obj, Tensor):
            return self._apply_image(obj, params)
        return obj

    def __call__(self, inputs: object) -> object:
        """Transform ``inputs`` — a tensor, a typed target, or a nested sample."""
        ref = _find_reference(inputs)
        if ref is None:
            raise ValueError(
                f"{type(self).__name__}: no image / mask found in the sample to "
                "derive transform parameters from"
            )
        params = self.make_params(ref)
        return self._dispatch(inputs, params)


class Compose(Transform):
    """Chain several transforms into a single callable.

    Works on a single image tensor or a multi-target sample alike — each
    transform's output (whatever its structure) feeds the next.

    Parameters
    ----------
    transforms : list of Transform
        Applied left-to-right.

    Examples
    --------
    >>> from lucid.utils.transforms import Compose, Resize, CenterCrop, Normalize
    >>> tf = Compose([
    ...     Resize(256),
    ...     CenterCrop(224),
    ...     Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ... ])
    >>> y = tf(image)            # image: lucid.Tensor (3, H, W) in [0, 1]
    """

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = list(transforms)

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
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
