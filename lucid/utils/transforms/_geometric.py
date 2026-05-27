"""Geometric transforms — spatial resampling / cropping.

Phase 1 ships the deterministic inference transforms (:class:`Resize`,
:class:`CenterCrop`).  Randomized augmentations (RandomResizedCrop,
RandomCrop, flips, Pad) arrive in the augmentation phase.
"""

from lucid._tensor import Tensor
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Transform


class Resize(Transform):
    r"""Resize an image to ``size``.

    Parameters
    ----------
    size : int or (int, int)
        If an ``int``, the shorter side is scaled to ``size`` with the
        aspect ratio preserved; if ``(h, w)``, resized to exactly that.
    interpolation : str, optional, default="bilinear"
        Interpolation mode (see :func:`lucid.nn.functional.interpolate`).

    Examples
    --------
    >>> Resize(256)(image).shape           # shorter side -> 256
    >>> Resize((224, 224))(image).shape    # exact
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        interpolation: str = "bilinear",
    ) -> None:
        self.size = size
        self.interpolation = interpolation

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return F.resize(img, self.size, interpolation=self.interpolation)

    def __repr__(self) -> str:
        return f"Resize(size={self.size}, interpolation={self.interpolation!r})"


class CenterCrop(Transform):
    r"""Crop a centered square (or ``(h, w)``) window.

    Parameters
    ----------
    size : int or (int, int)
        Output crop size; square if an ``int``.

    Examples
    --------
    >>> CenterCrop(224)(image).shape[-2:]
    (224, 224)
    """

    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = size

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return F.center_crop(img, self.size)

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"
