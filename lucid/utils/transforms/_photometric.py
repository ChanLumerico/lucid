"""Photometric transforms — per-pixel value adjustment.

Deterministic inference transforms (:class:`Normalize`,
:class:`Rescale`) plus stochastic augmentations (:class:`ColorJitter`,
:class:`RandomErasing`).
"""

from typing import cast

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import functional as F
from lucid.utils.transforms import _random
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


def _jitter_range(
    value: float | tuple[float, float],
    *,
    center: float,
    floor: float | None = None,
) -> tuple[float, float] | None:
    """Normalize a ColorJitter arg to a ``(min, max)`` range (or None).

    A scalar ``v`` becomes ``[center - v, center + v]`` (clamped at
    ``floor`` when given); a 2-tuple is used as-is.  ``0`` (or a
    no-op range) returns ``None`` so the factor is skipped.
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


class ColorJitter(Transform):
    r"""Randomly jitter brightness, contrast, saturation, and hue.

    Mirrors ``torchvision.transforms.ColorJitter``: each factor is
    sampled uniformly from its range and the four adjustments are
    applied in a random order.

    Parameters
    ----------
    brightness, contrast, saturation : float or (float, float), optional
        A scalar ``v`` gives the range ``[max(0, 1-v), 1+v]``; a tuple
        is used directly.  ``0`` disables that adjustment.
    hue : float or (float, float), optional
        A scalar ``v`` gives ``[-v, v]`` (``v`` ≤ 0.5); a tuple is used
        directly.  ``0`` disables hue jitter.

    Notes
    -----
    Operates on RGB images in ``[0, 1]``.  Hue uses an HSV round-trip
    (:func:`lucid.utils.transforms.functional.adjust_hue`).
    """

    def __init__(
        self,
        brightness: float | tuple[float, float] = 0.0,
        contrast: float | tuple[float, float] = 0.0,
        saturation: float | tuple[float, float] = 0.0,
        hue: float | tuple[float, float] = 0.0,
    ) -> None:
        self.brightness = _jitter_range(brightness, center=1.0, floor=0.0)
        self.contrast = _jitter_range(contrast, center=1.0, floor=0.0)
        self.saturation = _jitter_range(saturation, center=1.0, floor=0.0)
        self.hue = _jitter_range(hue, center=0.0)

    def make_params(self, img: Tensor) -> dict[str, object]:
        def _sample(rng: tuple[float, float] | None) -> float | None:
            return None if rng is None else _random.uniform(rng[0], rng[1])

        order = [0, 1, 2, 3]
        # Fisher-Yates using the Lucid RNG.
        for i in range(len(order) - 1, 0, -1):
            j = _random.randint(0, i + 1)
            order[i], order[j] = order[j], order[i]
        return {
            "order": order,
            "brightness": _sample(self.brightness),
            "contrast": _sample(self.contrast),
            "saturation": _sample(self.saturation),
            "hue": _sample(self.hue),
        }

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        order = cast(list[int], params["order"])
        for idx in order:
            if idx == 0 and params["brightness"] is not None:
                img = F.adjust_brightness(img, cast(float, params["brightness"]))
            elif idx == 1 and params["contrast"] is not None:
                img = F.adjust_contrast(img, cast(float, params["contrast"]))
            elif idx == 2 and params["saturation"] is not None:
                img = F.adjust_saturation(img, cast(float, params["saturation"]))
            elif idx == 3 and params["hue"] is not None:
                img = F.adjust_hue(img, cast(float, params["hue"]))
        return img

    def __repr__(self) -> str:
        return (
            f"ColorJitter(brightness={self.brightness}, contrast={self.contrast}, "
            f"saturation={self.saturation}, hue={self.hue})"
        )


class RandomErasing(Transform):
    r"""Randomly erase a rectangular region (Zhong et al., 2017).

    With probability ``p``, a region covering ``scale`` fraction of the
    image area with aspect ratio in ``ratio`` is filled with ``value``.

    Parameters
    ----------
    p : float, optional, default=0.5
        Probability of erasing.
    scale : (float, float), optional, default=(0.02, 0.33)
        Range of erased-area fraction.
    ratio : (float, float), optional, default=(0.3, 3.3)
        Range of erased-region aspect ratios.
    value : float, optional, default=0.0
        Fill value.

    Notes
    -----
    Expects a normalized or ``[0, 1]`` float image.  The erase is
    implemented as a multiplicative keep-mask (no in-place assignment).
    """

    def __init__(
        self,
        p: float = 0.5,
        *,
        scale: tuple[float, float] = (0.02, 0.33),
        ratio: tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
    ) -> None:
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def make_params(self, img: Tensor) -> dict[str, object]:
        import math

        if _random.rand() >= self.p:
            return {"erase": False}
        h, w = F._spatial_hw(img)
        area = float(h * w)
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = _random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(_random.uniform(log_ratio[0], log_ratio[1]))
            eh = int(round(math.sqrt(target_area / aspect)))
            ew = int(round(math.sqrt(target_area * aspect)))
            if 0 < eh <= h and 0 < ew <= w:
                top = _random.randint(0, h - eh + 1)
                left = _random.randint(0, w - ew + 1)
                return {
                    "erase": True,
                    "top": top,
                    "left": left,
                    "height": eh,
                    "width": ew,
                }
        return {"erase": False}

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        if not params["erase"]:
            return img
        h, w = F._spatial_hw(img)
        c = int(img.shape[-3])
        top = cast(int, params["top"])
        left = cast(int, params["left"])
        eh = cast(int, params["height"])
        ew = cast(int, params["width"])
        # Build a keep-mask: zeros inside the erased box, ones outside,
        # via padding a zero block with a ones border.
        inner = lucid.zeros(1, eh, ew, dtype=img.dtype)
        keep = F.pad(
            inner,
            (left, w - left - ew, top, h - top - eh),
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
