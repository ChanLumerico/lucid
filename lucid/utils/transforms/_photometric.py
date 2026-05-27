"""Photometric transforms — per-pixel value adjustment.

Deterministic inference transforms (:class:`Normalize`,
:class:`Rescale`) and stochastic augmentations (:class:`ColorJitter`,
:class:`RandomErasing`).  All are
:class:`~lucid.utils.transforms._base.PhotometricTransform` — they act
only on the image and leave masks / boxes untouched.
"""

import math
from dataclasses import dataclass

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import (
    Empty,
    PhotometricTransform,
    _NoParams,
    _ProbabilityGate,
)


# ── parameter types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ColorJitterParams:
    """Sampled per-call jitter factors + the order to apply them in."""

    order: tuple[int, ...]
    brightness: float | None
    contrast: float | None
    saturation: float | None
    hue: float | None


@dataclass(frozen=True)
class EraseParams:
    """Whether to erase this call, and (if so) the region to blank."""

    apply: bool
    top: int = 0
    left: int = 0
    height: int = 0
    width: int = 0


# ── deterministic ───────────────────────────────────────────────────


class Normalize(_NoParams, PhotometricTransform[Empty]):
    r"""Normalize an image per channel: ``(img - mean) / std``."""

    def __init__(self, mean: tuple[float, ...], std: tuple[float, ...]) -> None:
        self.mean = tuple(mean)
        self.std = tuple(std)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.normalize(img, self.mean, self.std)

    def __repr__(self) -> str:
        return f"Normalize(mean={self.mean}, std={self.std})"


class Rescale(_NoParams, PhotometricTransform[Empty]):
    r"""Scale pixel values by a constant (e.g. uint8 ``[0,255]`` → ``[0,1]``)."""

    def __init__(self, scale: float = 1.0 / 255.0) -> None:
        self.scale = scale

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.rescale(img, self.scale)

    def __repr__(self) -> str:
        return f"Rescale(scale={self.scale})"


# ── randomized ──────────────────────────────────────────────────────


def _jitter_range(
    value: float | tuple[float, float],
    *,
    center: float,
    floor: float | None = None,
) -> tuple[float, float] | None:
    """Normalize a ColorJitter arg to a ``(min, max)`` range, or ``None``.

    A scalar ``v`` → ``[center - v, center + v]`` (clamped at ``floor``);
    a 2-tuple is used directly; ``0`` → ``None`` (factor skipped).
    """
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
    r"""Randomly jitter brightness, contrast, saturation, and hue.

    Each factor is sampled uniformly from its range and the four
    adjustments are applied in a random order (matching torchvision).

    Parameters
    ----------
    brightness, contrast, saturation : float or (float, float), optional
        Scalar ``v`` → range ``[max(0, 1-v), 1+v]``; tuple used directly;
        ``0`` disables.
    hue : float or (float, float), optional
        Scalar ``v`` → ``[-v, v]`` (``v`` ≤ 0.5); ``0`` disables.
    """

    _ADJUST = (
        F.adjust_brightness,
        F.adjust_contrast,
        F.adjust_saturation,
        F.adjust_hue,
    )

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0.0,
        contrast: float | tuple[float, float] = 0.0,
        saturation: float | tuple[float, float] = 0.0,
        hue: float | tuple[float, float] = 0.0,
    ) -> None:
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
        factors = (
            params.brightness,
            params.contrast,
            params.saturation,
            params.hue,
        )
        for idx in params.order:
            factor = factors[idx]
            if factor is not None:
                img = self._ADJUST[idx](img, factor)
        return img

    def __repr__(self) -> str:
        b, c, s, h = self._ranges
        return f"ColorJitter(brightness={b}, contrast={c}, saturation={s}, hue={h})"


class RandomErasing(_ProbabilityGate, PhotometricTransform[EraseParams]):
    r"""Randomly erase a rectangular region (Zhong et al., 2017).

    With probability ``p`` a region covering ``scale`` of the area with
    aspect ratio in ``ratio`` is filled with ``value``.

    Parameters
    ----------
    p : float, optional, default=0.5
        Erase probability.
    scale : (float, float), optional, default=(0.02, 0.33)
        Range of erased-area fraction.
    ratio : (float, float), optional, default=(0.3, 3.3)
        Range of erased-region aspect ratios.
    value : float, optional, default=0.0
        Fill value.

    Notes
    -----
    Implemented as a multiplicative keep-mask (no in-place assignment).
    """

    def __init__(
        self,
        p: float = 0.5,
        *,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
    ) -> None:
        super().__init__(p)
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def make_params(self, img: Tensor) -> EraseParams:
        if not self._gate():
            return EraseParams(apply=False)
        h, w = F._spatial_hw(img)
        area = float(h * w)
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = _random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(_random.uniform(log_ratio[0], log_ratio[1]))
            eh = int(round(math.sqrt(target_area / aspect)))
            ew = int(round(math.sqrt(target_area * aspect)))
            if 0 < eh <= h and 0 < ew <= w:
                return EraseParams(
                    apply=True,
                    top=_random.randint(0, h - eh + 1),
                    left=_random.randint(0, w - ew + 1),
                    height=eh,
                    width=ew,
                )
        return EraseParams(apply=False)

    def _apply_image(self, img: Tensor, params: EraseParams) -> Tensor:
        if not params.apply:
            return img
        h, w = F._spatial_hw(img)
        c = int(img.shape[-3])
        inner = lucid.zeros(1, params.height, params.width, dtype=img.dtype)
        keep = F.pad(
            inner,
            (
                params.left,
                w - params.left - params.width,
                params.top,
                h - params.top - params.height,
            ),
            mode="constant",
            value=1.0,
        )
        keep = lucid.concat([keep] * c, dim=-3)  # type: ignore[arg-type]
        if img.ndim == 4:
            keep = keep[None]
        return img * keep + self.value * (1.0 - keep)

    def __repr__(self) -> str:
        return (
            f"RandomErasing(p={self.p}, scale={self.scale}, ratio={self.ratio}, "
            f"value={self.value})"
        )
