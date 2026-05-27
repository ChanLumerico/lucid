"""Photometric transforms — per-pixel value adjustment.

Phase 1 ships the deterministic inference transforms
(:class:`Normalize`, :class:`Rescale`).  Stochastic photometric
augmentations (ColorJitter, RandomErasing) arrive in the augmentation
phase.
"""

from lucid._tensor import Tensor
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Transform


class Normalize(Transform):
    r"""Normalize an image per channel: ``(img - mean) / std``.

    Parameters
    ----------
    mean : tuple of float
        Per-channel means (length = channel count).
    std : tuple of float
        Per-channel standard deviations.

    Examples
    --------
    >>> Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image)
    """

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.mean = tuple(mean)
        self.std = tuple(std)

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return F.normalize(img, self.mean, self.std)

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


class Rescale(Transform):
    r"""Scale pixel values by a constant (e.g. uint8 ``[0,255]`` → ``[0,1]``).

    Parameters
    ----------
    scale : float, optional, default=1/255
        Multiplier applied to every pixel.

    Examples
    --------
    >>> Rescale()(uint8_image)        # -> float in [0, 1]
    """

    def __init__(self, scale: float = 1.0 / 255.0) -> None:
        self.scale = scale

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return F.rescale(img, self.scale)

    def __repr__(self) -> str:
        return f"Rescale(scale={self.scale})"
