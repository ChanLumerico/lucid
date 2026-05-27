"""Photometric transforms — per-pixel value adjustment.

Albumentations-compatible colour / intensity transforms.  All are
:class:`~lucid.utils.transforms._base.PhotometricTransform` — they act
only on the image and leave masks / boxes / keypoints untouched.

This module currently holds :class:`Normalize` and :class:`ColorJitter`;
the wider Albumentations colour set (brightness/contrast, gamma, HSV,
channel ops, CLAHE, …) lands in later batches.
"""

from dataclasses import dataclass

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, PhotometricTransform, _NoParams


# ── parameter types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ColorJitterParams:
    """Sampled per-call jitter factors + the order to apply them in."""

    order: tuple[int, ...]
    brightness: float | None
    contrast: float | None
    saturation: float | None
    hue: float | None


# ── deterministic ───────────────────────────────────────────────────


class Normalize(_NoParams, PhotometricTransform[Empty]):
    r"""Normalize an image (Albumentations ``Normalize``).

    Computes ``(img - mean * max_pixel_value) / (std * max_pixel_value)``
    — i.e. scales by ``max_pixel_value`` then standardizes per channel.

    Parameters
    ----------
    mean, std : tuple of float
        Per-channel statistics.
    max_pixel_value : float, optional, default=255.0
        Value the inputs are divided by (``1.0`` for ``[0, 1]`` inputs).
    p : float, optional, default=1.0
    """

    def __init__(
        self,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        max_pixel_value: float = 255.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.mean = tuple(mean)
        self.std = tuple(std)
        self.max_pixel_value = max_pixel_value

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        mv = self.max_pixel_value
        eff_mean = tuple(m * mv for m in self.mean)
        eff_std = tuple(s * mv for s in self.std)
        return F.normalize(img, eff_mean, eff_std)

    def __repr__(self) -> str:
        return (
            f"Normalize(mean={self.mean}, std={self.std}, "
            f"max_pixel_value={self.max_pixel_value}, p={self.p})"
        )


# ── randomized ──────────────────────────────────────────────────────


def _jitter_range(
    value: float | tuple[float, float],
    *,
    center: float,
    floor: float | None = None,
) -> tuple[float, float] | None:
    """Normalize a ColorJitter arg to a ``(min, max)`` range, or ``None``."""
    if isinstance(value, (int, float)):
        if value == 0:
            return None
        lo, hi = center - value, center + value
    else:
        lo, hi = value[0], value[1]
    if floor is not None:
        lo = max(lo, floor)
    return (lo, hi)


class ColorJitter(PhotometricTransform[ColorJitterParams]):
    r"""Randomly jitter brightness/contrast/saturation/hue.

    Albumentations ``ColorJitter`` — each factor is sampled uniformly
    from its range and the four adjustments are applied in random order.

    Parameters
    ----------
    brightness, contrast, saturation : float or (float, float), optional, default=0.2
        Scalar ``v`` → range ``[max(0, 1-v), 1+v]``; tuple used directly.
    hue : float or (float, float), optional, default=0.2
        Scalar ``v`` → ``[-v, v]`` (``v`` ≤ 0.5).
    p : float, optional, default=0.5
    """

    _ADJUST = (
        F.adjust_brightness,
        F.adjust_contrast,
        F.adjust_saturation,
        F.adjust_hue,
    )

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0.2,
        contrast: float | tuple[float, float] = 0.2,
        saturation: float | tuple[float, float] = 0.2,
        hue: float | tuple[float, float] = 0.2,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self._ranges = (
            _jitter_range(brightness, center=1.0, floor=0.0),
            _jitter_range(contrast, center=1.0, floor=0.0),
            _jitter_range(saturation, center=1.0, floor=0.0),
            _jitter_range(hue, center=0.0),
        )

    def make_params(self, img: Tensor) -> ColorJitterParams:
        factors = [
            None if rng is None else _random.uniform(rng[0], rng[1])
            for rng in self._ranges
        ]
        order = list(range(4))
        for i in range(len(order) - 1, 0, -1):  # Fisher-Yates via Lucid RNG
            j = _random.randint(0, i + 1)
            order[i], order[j] = order[j], order[i]
        return ColorJitterParams(
            order=tuple(order),
            brightness=factors[0],
            contrast=factors[1],
            saturation=factors[2],
            hue=factors[3],
        )

    def _apply_image(self, img: Tensor, params: ColorJitterParams) -> Tensor:
        factors = (params.brightness, params.contrast, params.saturation, params.hue)
        for idx in params.order:
            factor = factors[idx]
            if factor is not None:
                img = self._ADJUST[idx](img, factor)
        return img

    def __repr__(self) -> str:
        b, c, s, h = self._ranges
        return (
            f"ColorJitter(brightness={b}, contrast={c}, saturation={s}, "
            f"hue={h}, p={self.p})"
        )
