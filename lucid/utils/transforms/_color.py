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
    r"""Random hue / saturation / value shift (Albumentations ``HueSaturationValue``).

    Samples each shift uniformly from its limit range, then applies a
    single :func:`functional.adjust_hsv` round-trip — additive in HSV,
    hue wraps, saturation and value clip to ``[0, 1]``.  Channel-gated
    to RGB (:meth:`PhotometricTransform._require_channels` rejects
    non-3-channel input).

    Parameters
    ----------
    hue_shift_limit : float or (float, float), optional, default=20
        Hue offset range on the OpenCV scale ``[0, 179]``.  Scalar
        ``v`` is interpreted as ``(-v, v)``.
    sat_shift_limit : float or (float, float), optional, default=30
        Saturation offset range on the OpenCV scale ``[0, 255]``.
    val_shift_limit : float or (float, float), optional, default=20
        Value (brightness) offset range on the OpenCV scale ``[0, 255]``.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If the image has anything other than 3 channels.

    Notes
    -----
    Lucid stays in float for the full HSV round-trip; Albumentations
    quantises through ``uint8`` + cv2 HSV, so the two match to
    ``~0.02`` (G3 parity "ballpark" tier).  Lucid's path is the more
    precise of the two.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.HueSaturationValue(hue_shift_limit=10, p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> tuple(out.shape)
    (3, 32, 32)
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
        self._require_channels(img, 3)
        return F.adjust_hsv(img, params.a, params.b, params.c)

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
    r"""Contrast-limited adaptive histogram equalization (Albumentations ``CLAHE``).

    Thin wrapper around :func:`functional.clahe`: per-tile
    clipped-histogram LUT + 4-neighbour bilinear interpolation between
    tile centres.  Single-channel images are equalized directly; RGB
    is routed through the HSV value channel so hue and saturation are
    preserved (mirrors Albumentations' "luminance-only" behaviour).

    Parameters
    ----------
    clip_limit : float, optional, default=4.0
        Contrast-clip threshold; per-tile histogram cap is
        ``clip_limit * tile_area / 256``.  Higher values allow more
        contrast amplification at the cost of noise.
    tile_grid_size : (int, int), optional, default=(8, 8)
        Number of tiles ``(rows, cols)`` the image is divided into.
        Larger grids give more local adaptivity at a per-tile cost.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Notes
    -----
    Approximate cv2 / Albumentations parity (colour-space pivot
    differs: HSV value vs LAB L).  Currently ~30× slower than Albu
    on a 224² RGB input — see :func:`functional.clahe` notes for the
    bottleneck description.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 64, 64))).data
    >>> tuple(out.shape)
    (3, 64, 64)
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
        return F.clahe(img, self.clip_limit, self.tile_grid_size)

    def __repr__(self) -> str:
        return (
            f"CLAHE(clip_limit={self.clip_limit}, "
            f"tile_grid_size={self.tile_grid_size}, p={self.p})"
        )


# ── tone / inversion ────────────────────────────────────────────────


class Solarize(PhotometricTransform[ScalarParams]):
    r"""Invert pixels above ``threshold`` (Albumentations ``Solarize``)."""

    def __init__(
        self, threshold: float | tuple[float, float] = 128, p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.threshold = (
            (threshold, threshold) if isinstance(threshold, (int, float)) else threshold
        )

    def make_params(self, img: Tensor) -> ScalarParams:
        return ScalarParams(value=_random.uniform(*self.threshold) / 255.0)

    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.where(img >= params.value, 1.0 - img, img)

    def __repr__(self) -> str:
        return f"Solarize(threshold={self.threshold}, p={self.p})"


class Posterize(PhotometricTransform[Empty]):
    r"""Reduce bits per channel (Albumentations ``Posterize``).

    Two quantisation modes — pick based on whether you need
    bit-exact Albumentations / cv2 parity or just a cheap float-only
    posterise.

    Parameters
    ----------
    num_bits : int, optional, default=4
        Number of bits to keep per channel; output has ``2**num_bits``
        distinct levels.  Must be in ``[1, 7]``.
    mode : {"uint8_mask", "float"}, optional, default="uint8_mask"
        Quantisation algorithm:

        * ``"uint8_mask"`` — matches Albumentations / OpenCV
          bit-exactly.  Round-trips through ``uint8`` and applies the
          bit mask ``~((1 << (8 - num_bits)) - 1)``, then divides by
          ``255``.  Use this for parity with reference pipelines.
        * ``"float"`` — pure-float quantisation via
          ``floor(x * 2**num_bits) / 2**num_bits``.  Slightly
          different mid-bin placement than the OpenCV bit-mask;
          cheaper (no uint8 round-trip).  Use for novel pipelines
          where Albu parity isn't required.
    p : float, optional, default=0.5
        Probability of applying the transform.

    Raises
    ------
    ValueError
        If ``num_bits`` falls outside ``[1, 7]`` or ``mode`` is not one
        of the two recognised names.

    Notes
    -----
    The G3 parity suite verifies the ``"uint8_mask"`` mode matches
    Albumentations to float32 epsilon across ``num_bits ∈ {1, …, 7}``.
    A 200-seed statistical aggregate (G4e) confirms the mean-of-max
    diff stays in the same regime.

    Examples
    --------
    >>> import lucid
    >>> import lucid.utils.transforms as T
    >>> tf = T.Posterize(num_bits=3, mode="uint8_mask", p=1.0)
    >>> out = tf(T.Image(lucid.rand(3, 32, 32))).data
    >>> # 3-bit → 8 distinct quantisation levels per channel
    >>> len(set(out.numpy().reshape(-1).tolist())) <= 8
    True
    """

    def __init__(
        self,
        num_bits: int = 4,
        mode: str = "uint8_mask",
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        if not 1 <= num_bits <= 7:
            raise ValueError(f"num_bits must be in [1, 7], got {num_bits}")
        if mode not in ("uint8_mask", "float"):
            raise ValueError(f"mode must be 'uint8_mask' or 'float', got {mode!r}")
        self.num_bits = num_bits
        self.mode = mode

    def make_params(self, img: Tensor) -> Empty:
        from lucid.utils.transforms._base import NO_PARAMS

        return NO_PARAMS

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        if self.mode == "float":
            levels = float(2**self.num_bits)
            return lucid.floor(lucid.clip(img, 0.0, 1.0) * levels) / levels
        # uint8_mask — round to uint8, apply bit mask, back to float
        mask_int = (~((1 << (8 - self.num_bits)) - 1)) & 0xFF
        u8 = lucid.clip(lucid.round(img * 255.0), 0.0, 255.0).long()
        masked = u8 & mask_int
        return masked.to(img.dtype) / 255.0

    def __repr__(self) -> str:
        return f"Posterize(num_bits={self.num_bits}, mode={self.mode!r}, p={self.p})"


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


# ── B8: additional colour / pixel transforms ────────────────────────


class RandomBrightness(PhotometricTransform[ScalarParams]):
    r"""Random brightness only (Albumentations ``RandomBrightness``)."""

    def __init__(
        self, limit: float | tuple[float, float] = 0.2, p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.limit = _rng(limit)

    def make_params(self, img: Tensor) -> ScalarParams:
        return ScalarParams(value=_random.uniform(*self.limit))

    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.clip(img + params.value, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"RandomBrightness(limit={self.limit}, p={self.p})"


class RandomContrast(PhotometricTransform[ScalarParams]):
    r"""Random contrast only (Albumentations ``RandomContrast``)."""

    def __init__(
        self, limit: float | tuple[float, float] = 0.2, p: float = 0.5
    ) -> None:
        super().__init__(p=p)
        self.limit = _rng(limit)

    def make_params(self, img: Tensor) -> ScalarParams:
        return ScalarParams(value=_random.uniform(*self.limit))

    def _apply_image(self, img: Tensor, params: ScalarParams) -> Tensor:
        return lucid.clip(img * (1.0 + params.value), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"RandomContrast(limit={self.limit}, p={self.p})"


class UnsharpMask(PhotometricTransform[TripletParams]):
    r"""Unsharp masking — sharpen via ``img + alpha*(img - blur)`` (Albu ``UnsharpMask``)."""

    def __init__(
        self,
        blur_limit: tuple[int, int] = (3, 7),
        sigma_limit: float | tuple[float, float] = 0.0,
        alpha: tuple[float, float] = (0.2, 0.5),
        threshold: float = 10.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit
        self.sigma_limit = (
            (0.0, sigma_limit) if isinstance(sigma_limit, (int, float)) else sigma_limit
        )
        self.alpha = alpha
        self.threshold = threshold

    def make_params(self, img: Tensor) -> TripletParams:
        from lucid.utils.transforms._blur import _odd

        k = _odd(_random.randint(self.blur_limit[0], self.blur_limit[1] + 1))
        sigma = _random.uniform(self.sigma_limit[0], self.sigma_limit[1])
        if sigma <= 0.0:
            sigma = 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8
        return TripletParams(a=_random.uniform(*self.alpha), b=float(k), c=sigma)

    def _apply_image(self, img: Tensor, params: TripletParams) -> Tensor:
        blur = F.gaussian_blur(img, params.c, ksize=int(params.b))
        return lucid.clip(img + params.a * (img - blur), 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"UnsharpMask(blur_limit={self.blur_limit}, alpha={self.alpha}, p={self.p})"
        )


class RingingOvershoot(PhotometricTransform[Empty]):
    r"""Ringing overshoot via a high-pass kernel blend (Albu ``RingingOvershoot``)."""

    def __init__(self, blur_limit: tuple[int, int] = (7, 15), p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    def make_params(self, img: Tensor) -> Empty:
        from lucid.utils.transforms._base import NO_PARAMS

        return NO_PARAMS

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        # Approximate ringing with a sharpening (high-pass) kernel.
        kernel = [[0.0, -0.25, 0.0], [-0.25, 2.0, -0.25], [0.0, -0.25, 0.0]]
        return lucid.clip(F.depthwise_conv2d(img, kernel), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"RingingOvershoot(blur_limit={self.blur_limit}, p={self.p})"


class FancyPCA(PhotometricTransform[Empty]):
    r"""AlexNet-style PCA colour augmentation (Albumentations ``FancyPCA``)."""

    def __init__(self, alpha: float = 0.1, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.alpha = alpha

    def make_params(self, img: Tensor) -> Empty:
        from lucid.utils.transforms._base import NO_PARAMS

        return NO_PARAMS

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        self._require_channels(img, 3)
        c = 3
        flat = img.reshape(c, -1)  # (3, N)
        mean = lucid.mean(flat, dim=[1], keepdim=True)
        centered = flat - mean
        cov = lucid.matmul(centered, lucid.swapaxes(centered, 0, 1)) / float(flat.shape[1])  # type: ignore[arg-type]
        evals, evecs = lucid.linalg.eigh(cov)
        alphas = [self.alpha * _random.uniform(-1.0, 1.0) for _ in range(c)]
        scaled = lucid.tensor([alphas[i] for i in range(c)], dtype=img.dtype).reshape(
            c, 1
        ) * evals.reshape(c, 1)
        delta = lucid.matmul(evecs, scaled).reshape(c, 1, 1)  # (3,1,1)
        if img.ndim == 4:
            delta = delta[None]
        return lucid.clip(img + delta, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"FancyPCA(alpha={self.alpha}, p={self.p})"


@dataclass(frozen=True)
class PixelMaskParams:
    mask: Tensor


class PixelDropout(PhotometricTransform[PixelMaskParams]):
    r"""Randomly set pixels to ``drop_value`` (Albumentations ``PixelDropout``)."""

    def __init__(
        self,
        dropout_prob: float = 0.01,
        per_channel: bool = False,
        drop_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.dropout_prob = dropout_prob
        self.per_channel = per_channel
        self.drop_value = drop_value

    def make_params(self, img: Tensor) -> PixelMaskParams:
        c, h, w = int(img.shape[-3]), *F._spatial_hw(img)
        shape = (c, h, w) if self.per_channel else (1, h, w)
        keep = (lucid.rand(*shape) >= self.dropout_prob).to(img.dtype)
        return PixelMaskParams(mask=keep)

    def _apply_image(self, img: Tensor, params: PixelMaskParams) -> Tensor:
        keep = params.mask[None] if img.ndim == 4 else params.mask
        return img * keep + self.drop_value * (1.0 - keep)

    def __repr__(self) -> str:
        return f"PixelDropout(dropout_prob={self.dropout_prob}, p={self.p})"


@dataclass(frozen=True)
class BandParams:
    rows: tuple[tuple[int, int], ...]
    cols: tuple[tuple[int, int], ...]


class XYMasking(PhotometricTransform[BandParams]):
    r"""Mask random horizontal + vertical bands (Albumentations ``XYMasking``)."""

    def __init__(
        self,
        num_masks_x: int = 0,
        num_masks_y: int = 0,
        mask_x_length: int = 10,
        mask_y_length: int = 10,
        fill_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.num_masks_x = num_masks_x
        self.num_masks_y = num_masks_y
        self.mask_x_length = mask_x_length
        self.mask_y_length = mask_y_length
        self.fill_value = fill_value

    def make_params(self, img: Tensor) -> BandParams:
        h, w = F._spatial_hw(img)
        cols = tuple(
            (lambda s: (s, min(s + self.mask_x_length, w)))(
                _random.randint(0, max(w - 1, 1))
            )
            for _ in range(self.num_masks_x)
        )
        rows = tuple(
            (lambda s: (s, min(s + self.mask_y_length, h)))(
                _random.randint(0, max(h - 1, 1))
            )
            for _ in range(self.num_masks_y)
        )
        return BandParams(rows=rows, cols=cols)

    def _apply_image(self, img: Tensor, params: BandParams) -> Tensor:
        h, w = F._spatial_hw(img)
        keep = lucid.ones(1, h, w, dtype=img.dtype)
        for r0, r1 in params.rows:
            band = F.pad(
                lucid.zeros(1, r1 - r0, w, dtype=img.dtype),
                (0, 0, r0, h - r1),
                value=1.0,
            )
            keep = keep * band
        for c0, c1 in params.cols:
            band = F.pad(
                lucid.zeros(1, h, c1 - c0, dtype=img.dtype),
                (c0, w - c1, 0, 0),
                value=1.0,
            )
            keep = keep * band
        keep_b = keep[None] if img.ndim == 4 else keep
        return img * keep_b + self.fill_value * (1.0 - keep_b)

    def __repr__(self) -> str:
        return (
            f"XYMasking(num_masks_x={self.num_masks_x}, "
            f"num_masks_y={self.num_masks_y}, p={self.p})"
        )
