"""Core transform primitives — :class:`Transform` ABC + :class:`Compose`.

Lucid's transform library follows the torchvision-v2 design adapted to
the Lucid-tensor-only world (no numpy / PIL inside ``lucid/``, per H4):
transforms consume and produce :class:`lucid.Tensor` images shaped
``(C, H, W)`` or ``(B, C, H, W)``.  Image *decoding* (file → tensor)
lives outside this package.

Forward-compatible dispatch
---------------------------
Each transform splits into two hooks:

* :meth:`Transform.make_params` — sample any randomness *once* per call
  (so a random crop/flip applies consistently across an image and its
  future mask / bounding-box companions).
* :meth:`Transform._apply_image` — apply the (possibly randomized)
  transform to an image tensor.

Phase 1 operates on plain image tensors.  The multi-target phase
(Image / Mask / BoundingBoxes) extends :meth:`__call__` to dispatch
each typed target to its own ``_apply_*`` hook using the same
``params`` — additive, no rewrite of existing transforms.
"""

import abc

from lucid._tensor import Tensor


class Transform(abc.ABC):
    """Abstract base for a single image transform.

    Subclasses implement :meth:`_apply_image` (required) and optionally
    :meth:`make_params` (for randomized transforms).  Calling the
    transform routes through both.
    """

    def make_params(self, img: Tensor) -> dict[str, object]:
        """Sample per-call parameters (randomness) for ``img``.

        Returns an empty dict for deterministic transforms.  Randomized
        transforms (e.g. ``RandomCrop``) override this to draw their
        parameters once, so the same parameters can later be reused
        across an image's mask / box companions.

        Parameters
        ----------
        img : Tensor
            The input image, used to size random parameters (e.g. crop
            offsets within the image's spatial extent).

        Returns
        -------
        dict
            Parameter bundle passed to :meth:`_apply_image`.
        """
        return {}

    @abc.abstractmethod
    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        """Apply the transform to an image tensor given sampled ``params``."""
        raise NotImplementedError

    def __call__(self, img: Tensor) -> Tensor:
        """Transform ``img`` (``(C, H, W)`` or ``(B, C, H, W)``)."""
        params = self.make_params(img)
        return self._apply_image(img, params)


class Compose(Transform):
    """Chain several transforms into a single callable.

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
    >>> y = tf(image)            # image: lucid.Tensor (3, H, W) in [0, 1]
    """

    def __init__(self, transforms: list[Transform]) -> None:
        self.transforms = list(transforms)

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        for tf in self.transforms:
            img = tf(img)
        return img

    def __call__(self, img: Tensor) -> Tensor:
        for tf in self.transforms:
            img = tf(img)
        return img

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.transforms)
        return f"Compose([{inner}])"
