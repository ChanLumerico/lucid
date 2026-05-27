"""Colour / pixel-level transforms (Albumentations-compatible).

All :class:`~lucid.utils.transforms._base.PhotometricTransform` — they
act only on the image (RGB ``[0, 1]``) and leave masks / boxes /
keypoints untouched.
"""

from dataclasses import dataclass

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, PhotometricTransform, _NoParams


def _rng(value: float | tuple[float, float]) -> tuple[float, float]:
    return (-value, value) if isinstance(value, (int, float)) else value


# ── parameter types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class BCParams:
    brightness: float
    contrast: float


@dataclass(frozen=True)
class ScalarParams:
    value: float


@dataclass(frozen=True)
class TripletParams:
    a: float
    b: float
    c: float


@dataclass(frozen=True)
class PermParams:
    order: tuple[int, int, int]


@dataclass(frozen=True)
class ChannelDropParams:
    channels: tuple[int, ...]


# ── brightness / contrast / gamma ───────────────────────────────────


class RandomBrightnessContrast(PhotometricTransform[BCParams]):
    r"""Random brightness + contrast (Albumentations ``RandomBrightnessContrast``)."""

    def __init__(
        self,
        brightness_limit: float | tuple[float, float] = 0.2,
        contrast_limit: float | tuple[float, float] = 0.2,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.brightness_limit = _rng(brightness_limit)
        self.contrast_limit = _rng(contrast_limit)

    def make_params(self, img: Tensor) -> BCParams:
        return BCParams(
            brightness=_random.uniform(*self.brightness_limit),
            contrast=_random.uniform(*self.contrast_limit),
        )

    def _apply_image(self, img: Tensor, params: BCParams) -> Tensor:
        out = img * (1.0 + params.contrast) + params.brightness
        return lucid.clip(out, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"RandomBrightnessContrast(brightness_limit={self.brightness_limit}, "
            f"contrast_limit={self.contrast_limit}, p={self.p})"
        )


class RandomGamma(PhotometricTransform[ScalarParams]):
    r"""Random gamma correction (Albumentations ``RandomGamma``)."""

    def __init__(
        self, gamma_limit: tuple[int, int] = (80, 120), p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.gamma_limit = gamma_limit

    def make_params(self, img: Tensor) -> ScalarParams:
        return ScalarParams(
            value=_random.uniform(self.gamma_limit[0], self.gamma_limit[1]) / 100.0
        )

    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.clip(img, 0.0, 1.0) ** params.value

    def __repr__(self) -> str:
        return f"RandomGamma(gamma_limit={self.gamma_limit}, p={self.p})"


# ── HSV / RGB shifts ────────────────────────────────────────────────


class HueSaturationValue(PhotometricTransform[TripletParams]):
    r"""Shift hue / saturation / value (Albumentations ``HueSaturationValue``).

    ``hue_shift_limit`` is in degrees (mapped to a fractional hue shift);
    saturation / value shifts are applied as multiplicative-ish blends.
    """

    def __init__(
        self,
        hue_shift_limit: float | tuple[float, float] = 20,
        sat_shift_limit: float | tuple[float, float] = 30,
        val_shift_limit: float | tuple[float, float] = 20,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.hue_shift_limit = _rng(hue_shift_limit)
        self.sat_shift_limit = _rng(sat_shift_limit)
        self.val_shift_limit = _rng(val_shift_limit)

    def make_params(self, img: Tensor) -> TripletParams:
        return TripletParams(
            a=_random.uniform(*self.hue_shift_limit),
            b=_random.uniform(*self.sat_shift_limit),
            c=_random.uniform(*self.val_shift_limit),
        )

    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        out = F.adjust_hue(img, params.a / 360.0)
        out = F.adjust_saturation(out, 1.0 + params.b / 100.0)
        out = F.adjust_brightness(out, 1.0 + params.c / 100.0)
        return out

    def __repr__(self) -> str:
        return (
            f"HueSaturationValue(hue_shift_limit={self.hue_shift_limit}, "
            f"sat_shift_limit={self.sat_shift_limit}, "
            f"val_shift_limit={self.val_shift_limit}, p={self.p})"
        )


class RGBShift(PhotometricTransform[TripletParams]):
    r"""Add per-channel shifts (Albumentations ``RGBShift``); limits on 0-255 scale."""

    def __init__(
        self,
        r_shift_limit: float | tuple[float, float] = 20,
        g_shift_limit: float | tuple[float, float] = 20,
        b_shift_limit: float | tuple[float, float] = 20,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.r = _rng(r_shift_limit)
        self.g = _rng(g_shift_limit)
        self.b = _rng(b_shift_limit)

    def make_params(self, img: Tensor) -> TripletParams:
        return TripletParams(
            a=_random.uniform(*self.r) / 255.0,
            b=_random.uniform(*self.g) / 255.0,
            c=_random.uniform(*self.b) / 255.0,
        )

    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        self._require_channels(img, 3)
        shift = lucid.tensor([params.a, params.b, params.c], dtype=img.dtype)
        shift = shift.reshape(1, 3, 1, 1) if img.ndim == 4 else shift.reshape(3, 1, 1)
        return lucid.clip(img + shift, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"RGBShift(r={self.r}, g={self.g}, b={self.b}, p={self.p})"


# ── channel ops ─────────────────────────────────────────────────────


class ChannelShuffle(PhotometricTransform[PermParams]):
    r"""Randomly permute the RGB channels (Albumentations ``ChannelShuffle``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def make_params(self, img: Tensor) -> PermParams:
        order = [0, 1, 2]
        for i in range(2, 0, -1):
            j = _random.randint(0, i + 1)
            order[i], order[j] = order[j], order[i]
        return PermParams(order=(order[0], order[1], order[2]))

    def _apply_image(self, img: Tensor, params: PermParams) -> Tensor:
        self._require_channels(img, 3)
        ax = -3
        ch = [img[..., i : i + 1, :, :] for i in params.order]
        return F._cat(ch, ax)

    def __repr__(self) -> str:
        return f"ChannelShuffle(p={self.p})"


class ChannelDropout(PhotometricTransform[ChannelDropParams]):
    r"""Zero out random channels (Albumentations ``ChannelDropout``)."""

    def __init__(
        self,
        channel_drop_range: tuple[int, int] = (1, 1),
        fill_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.channel_drop_range = channel_drop_range
        self.fill_value = fill_value

    def make_params(self, img: Tensor) -> ChannelDropParams:
        c = int(img.shape[-3])
        n = _random.randint(self.channel_drop_range[0], self.channel_drop_range[1] + 1)
        idxs = list(range(c))
        chosen: list[int] = []
        for _ in range(min(n, c)):
            k = _random.randint(0, len(idxs))
            chosen.append(idxs.pop(k))
        return ChannelDropParams(channels=tuple(chosen))

    def _apply_image(self, img: Tensor, params: ChannelDropParams) -> Tensor:
        c = int(img.shape[-3])
        keep = [0.0 if i in params.channels else 1.0 for i in range(c)]
        mask = lucid.tensor(keep, dtype=img.dtype)
        mask = mask.reshape(1, c, 1, 1) if img.ndim == 4 else mask.reshape(c, 1, 1)
        return img * mask + self.fill_value * (1.0 - mask)

    def __repr__(self) -> str:
        return (
            f"ChannelDropout(channel_drop_range={self.channel_drop_range}, p={self.p})"
        )


# ── histogram-based ─────────────────────────────────────────────────


class Equalize(_NoParams, PhotometricTransform[Empty]):
    r"""Per-channel histogram equalization (Albumentations ``Equalize``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.equalize(img)

    def __repr__(self) -> str:
        return f"Equalize(p={self.p})"


class CLAHE(_NoParams, PhotometricTransform[Empty]):
    r"""Contrast-limited histogram equalization (Albumentations ``CLAHE``).

    Note
    ----
    Implemented as a *global* clip-limited equalization (the contrast
    clipping of CLAHE without per-tile adaptivity).

    Parameters
    ----------
    clip_limit : float, optional, default=4.0
    tile_grid_size : (int, int), optional, default=(8, 8)
        Accepted for signature parity (unused in the global variant).
    p : float, optional, default=0.5
    """

    def __init__(
        self,
        clip_limit: float = 4.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.equalize(img, clip_limit=self.clip_limit)

    def __repr__(self) -> str:
        return f"CLAHE(clip_limit={self.clip_limit}, p={self.p})"


# ── tone / inversion ────────────────────────────────────────────────


class Solarize(PhotometricTransform[ScalarParams]):
    r"""Invert pixels above ``threshold`` (Albumentations ``Solarize``)."""

    def __init__(
        self, threshold: float | tuple[float, float] = 128, p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.threshold = (threshold, threshold) if isinstance(threshold, (int, float)) else threshold

    def make_params(self, img: Tensor) -> ScalarParams:
        return ScalarParams(value=_random.uniform(*self.threshold) / 255.0)

    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.where(img >= params.value, 1.0 - img, img)

    def __repr__(self) -> str:
        return f"Solarize(threshold={self.threshold}, p={self.p})"


class Posterize(PhotometricTransform[Empty]):
    r"""Reduce bits per channel (Albumentations ``Posterize``)."""

    def __init__(self, num_bits: int = 4, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.num_bits = num_bits

    def make_params(self, img: Tensor) -> Empty:
        from lucid.utils.transforms._base import NO_PARAMS

        return NO_PARAMS

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        levels = float(2 ** self.num_bits)
        return lucid.floor(lucid.clip(img, 0.0, 1.0) * levels) / levels

    def __repr__(self) -> str:
        return f"Posterize(num_bits={self.num_bits}, p={self.p})"


class InvertImg(_NoParams, PhotometricTransform[Empty]):
    r"""Invert intensities ``1 - img`` (Albumentations ``InvertImg``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return 1.0 - img

    def __repr__(self) -> str:
        return f"InvertImg(p={self.p})"


class ToGray(_NoParams, PhotometricTransform[Empty]):
    r"""Convert to grayscale, keeping 3 channels (Albumentations ``ToGray``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.rgb_to_grayscale(img, keep_channels=True)

    def __repr__(self) -> str:
        return f"ToGray(p={self.p})"


class ToSepia(_NoParams, PhotometricTransform[Empty]):
    r"""Apply a sepia colour matrix (Albumentations ``ToSepia``)."""

    _M = (
        (0.393, 0.769, 0.189),
        (0.349, 0.686, 0.168),
        (0.272, 0.534, 0.131),
    )

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        self._require_channels(img, 3)
        r = img[..., 0:1, :, :]
        g = img[..., 1:2, :, :]
        b = img[..., 2:3, :, :]
        m = self._M
        nr = m[0][0] * r + m[0][1] * g + m[0][2] * b
        ng = m[1][0] * r + m[1][1] * g + m[1][2] * b
        nb = m[2][0] * r + m[2][1] * g + m[2][2] * b
        return lucid.clip(F._cat([nr, ng, nb], -3), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"ToSepia(p={self.p})"


# ── sharpen / emboss ────────────────────────────────────────────────


class Sharpen(PhotometricTransform[TripletParams]):
    r"""Sharpen via unsharp kernel blend (Albumentations ``Sharpen``)."""

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        lightness: tuple[float, float] = (0.5, 1.0),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.alpha = alpha
        self.lightness = lightness

    def make_params(self, img: Tensor) -> TripletParams:
        return TripletParams(
            a=_random.uniform(*self.alpha),
            b=_random.uniform(*self.lightness),
            c=0.0,
        )

    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        light = params.b
        kernel = [[-1.0, -1.0, -1.0], [-1.0, 8.0 + light, -1.0], [-1.0, -1.0, -1.0]]
        sharp = F.depthwise_conv2d(img, kernel)
        out = (1.0 - params.a) * img + params.a * sharp
        return lucid.clip(out, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"Sharpen(alpha={self.alpha}, lightness={self.lightness}, p={self.p})"


class Emboss(PhotometricTransform[TripletParams]):
    r"""Emboss via directional kernel blend (Albumentations ``Emboss``)."""

    def __init__(
        self,
        alpha: tuple[float, float] = (0.2, 0.5),
        strength: tuple[float, float] = (0.2, 0.7),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.alpha = alpha
        self.strength = strength

    def make_params(self, img: Tensor) -> TripletParams:
        return TripletParams(
            a=_random.uniform(*self.alpha),
            b=_random.uniform(*self.strength),
            c=0.0,
        )

    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        s = params.b
        kernel = [[-1.0 - s, -s, 0.0], [-s, 1.0, s], [0.0, s, 1.0 + s]]
        emb = F.depthwise_conv2d(img, kernel)
        out = (1.0 - params.a) * img + params.a * emb
        return lucid.clip(out, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"Emboss(alpha={self.alpha}, strength={self.strength}, p={self.p})"


class RandomToneCurve(PhotometricTransform[ScalarParams]):
    r"""Random S-shaped tone curve (Albumentations ``RandomToneCurve``)."""

    def __init__(self, scale: float = 0.1, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.scale = scale

    def make_params(self, img: Tensor) -> ScalarParams:
        return ScalarParams(value=_random.uniform(-self.scale, self.scale))

    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        # Smooth S-curve around 0.5 controlled by the sampled amount.
        x = lucid.clip(img, 0.0, 1.0)
        out = x + params.value * lucid.sin(x * (2.0 * 3.141592653589793))
        return lucid.clip(out, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"RandomToneCurve(scale={self.scale}, p={self.p})"
