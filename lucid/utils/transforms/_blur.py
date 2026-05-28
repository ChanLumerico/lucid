"""Blur + noise transforms (Albumentations-compatible).

All :class:`~lucid.utils.transforms._base.PhotometricTransform` — they
act only on the image and leave masks / boxes / keypoints untouched.
"""

import math
from dataclasses import dataclass

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import PhotometricTransform


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


@dataclass(frozen=True)
class KSizeParam:
    ksize: int


@dataclass(frozen=True)
class SigmaParam:
    ksize: int
    sigma: float


@dataclass(frozen=True)
class MotionParam:
    ksize: int
    angle: float


@dataclass(frozen=True)
class NoiseParam:
    std: float


@dataclass(frozen=True)
class MultiplierParam:
    lo: float
    hi: float


@dataclass(frozen=True)
class ScaleParam:
    scale: float


# ── blur family ─────────────────────────────────────────────────────


class Blur(PhotometricTransform[KSizeParam]):
    r"""Box (mean) blur with a random odd kernel size (Albumentations ``Blur``).

    Samples an odd kernel size ``k`` uniformly from ``[3, blur_limit]``
    and convolves the image with a normalised ``k x k`` all-ones kernel
    via :func:`functional.depthwise_conv2d`.

    Parameters
    ----------
    blur_limit : int, optional, default=7
        Inclusive upper bound on the sampled kernel size; values are
        rounded up to the next odd integer to keep the kernel centred.
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(self, blur_limit: int = 7, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    def make_params(self, img: Tensor) -> KSizeParam:
        return KSizeParam(ksize=_odd(_random.randint(3, self.blur_limit + 1)))

    def _apply_image(self, img: Tensor, params: KSizeParam) -> Tensor:
        k = params.ksize
        kernel = [[1.0 / (k * k)] * k for _ in range(k)]
        return F.depthwise_conv2d(img, kernel)

    def __repr__(self) -> str:
        return f"Blur(blur_limit={self.blur_limit}, p={self.p})"


class MedianBlur(PhotometricTransform[KSizeParam]):
    r"""Median filter with a random odd kernel (Albumentations ``MedianBlur``).

    Samples an odd kernel size ``k`` uniformly from ``[3, blur_limit]``,
    builds the ``k * k`` shifted-neighbour stack via :func:`lucid.roll`,
    sorts along the stack axis, and returns the median element — an
    impulse-noise denoiser that preserves edges better than ``Blur``.

    Parameters
    ----------
    blur_limit : int, optional, default=7
        Inclusive upper bound on the sampled kernel size; rounded up
        to the next odd integer so the window is centred.
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(self, blur_limit: int = 7, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    def make_params(self, img: Tensor) -> KSizeParam:
        return KSizeParam(ksize=_odd(_random.randint(3, self.blur_limit + 1)))

    def _apply_image(self, img: Tensor, params: KSizeParam) -> Tensor:
        k = params.ksize
        half = k // 2
        neigh = [
            lucid.roll(img, (dy, dx), dims=(-2, -1))  # type: ignore[arg-type]
            for dy in range(-half, half + 1)
            for dx in range(-half, half + 1)
        ]
        stacked = lucid.stack(neigh, dim=0)
        ordered = lucid.sort(stacked, dim=0)
        return ordered[(k * k) // 2]

    def __repr__(self) -> str:
        return f"MedianBlur(blur_limit={self.blur_limit}, p={self.p})"


class MotionBlur(PhotometricTransform[MotionParam]):
    r"""Directional (linear streak) motion blur (Albumentations ``MotionBlur``).

    Samples an odd kernel size and a streak angle in ``[0, 180)°``, draws
    a 1-pixel-wide line through the kernel centre at that angle, then
    convolves the normalised kernel with the image — approximates the
    look of a camera panning while the shutter is open.

    Parameters
    ----------
    blur_limit : int, optional, default=7
        Inclusive upper bound on the kernel size; rounded up to the
        next odd integer.
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(self, blur_limit: int = 7, p: float = 0.5) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit

    def make_params(self, img: Tensor) -> MotionParam:
        return MotionParam(
            ksize=_odd(_random.randint(3, self.blur_limit + 1)),
            angle=_random.uniform(0.0, 180.0),
        )

    def _apply_image(self, img: Tensor, params: MotionParam) -> Tensor:
        k = params.ksize
        half = k // 2
        kernel = [[0.0] * k for _ in range(k)]
        rad = math.radians(params.angle)
        dx, dy = math.cos(rad), math.sin(rad)
        for t in range(-half, half + 1):
            x = int(round(half + t * dx))
            y = int(round(half + t * dy))
            if 0 <= x < k and 0 <= y < k:
                kernel[y][x] = 1.0
        total = sum(sum(r) for r in kernel) or 1.0
        kernel = [[v / total for v in r] for r in kernel]
        return F.depthwise_conv2d(img, kernel)

    def __repr__(self) -> str:
        return f"MotionBlur(blur_limit={self.blur_limit}, p={self.p})"


class GaussianBlur(PhotometricTransform[SigmaParam]):
    r"""Gaussian blur with random kernel size and sigma (Albumentations ``GaussianBlur``).

    Samples an odd kernel size from ``blur_limit`` and a sigma from
    ``sigma_limit``; when sigma is left at zero the OpenCV default
    ``0.3 * ((k - 1) * 0.5 - 1.0) + 0.8`` is used so the kernel matches
    cv2's auto-sigma policy.

    Parameters
    ----------
    blur_limit : (int, int), optional, default=(3, 7)
        Inclusive range for the kernel size; both ends are rounded up
        to the next odd integer.
    sigma_limit : float or (float, float), optional, default=0.0
        Range for the Gaussian sigma.  A scalar ``v`` expands to
        ``(0.0, v)``; ``0`` triggers the OpenCV auto-sigma formula.
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(
        self,
        blur_limit: tuple[int, int] = (3, 7),
        sigma_limit: float | tuple[float, float] = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.blur_limit = blur_limit
        self.sigma_limit = (
            (0.0, sigma_limit) if isinstance(sigma_limit, (int, float)) else sigma_limit
        )

    def make_params(self, img: Tensor) -> SigmaParam:
        k = _odd(_random.randint(self.blur_limit[0], self.blur_limit[1] + 1))
        sigma = _random.uniform(self.sigma_limit[0], self.sigma_limit[1])
        if sigma <= 0.0:
            sigma = 0.3 * ((k - 1) * 0.5 - 1.0) + 0.8  # OpenCV default
        return SigmaParam(ksize=k, sigma=sigma)

    def _apply_image(self, img: Tensor, params: SigmaParam) -> Tensor:
        return F.gaussian_blur(img, params.sigma, ksize=params.ksize)

    def __repr__(self) -> str:
        return f"GaussianBlur(blur_limit={self.blur_limit}, p={self.p})"


# ── noise family ────────────────────────────────────────────────────


class GaussNoise(PhotometricTransform[NoiseParam]):
    r"""Additive Gaussian noise (Albumentations ``GaussNoise``).

    ``var_limit`` is on the 0-255 scale (matching Albumentations).
    """

    def __init__(
        self,
        var_limit: tuple[float, float] = (10.0, 50.0),
        mean: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.var_limit = var_limit
        self.mean = mean

    def make_params(self, img: Tensor) -> NoiseParam:
        var = _random.uniform(self.var_limit[0], self.var_limit[1])
        return NoiseParam(std=math.sqrt(var) / 255.0)

    def _apply_image(self, img: Tensor, params: NoiseParam) -> Tensor:
        noise = lucid.randn(*img.shape) * params.std + self.mean / 255.0
        return lucid.clip(img + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"GaussNoise(var_limit={self.var_limit}, p={self.p})"


class MultiplicativeNoise(PhotometricTransform[MultiplierParam]):
    r"""Multiply pixel values by random noise (Albumentations ``MultiplicativeNoise``).

    With ``elementwise=False`` (default) a single scalar is sampled
    per call and the whole image is scaled by it; with
    ``elementwise=True`` an independent multiplier is sampled per
    pixel.  The output is always clipped back to ``[0, 1]``.

    Parameters
    ----------
    multiplier : (float, float), optional, default=(0.9, 1.1)
        Inclusive range of the per-pixel (or per-image) multiplier.
    elementwise : bool, optional, default=False
        If ``True``, sample one multiplier per pixel rather than one
        per image.
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(
        self,
        multiplier: tuple[float, float] = (0.9, 1.1),
        elementwise: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.multiplier = multiplier
        self.elementwise = elementwise

    def make_params(self, img: Tensor) -> MultiplierParam:
        return MultiplierParam(lo=self.multiplier[0], hi=self.multiplier[1])

    def _apply_image(self, img: Tensor, params: MultiplierParam) -> Tensor:
        if self.elementwise:
            mult = lucid.rand(*img.shape) * (params.hi - params.lo) + params.lo
            return lucid.clip(img * mult, 0.0, 1.0)
        scalar = _random.uniform(params.lo, params.hi)
        return lucid.clip(img * scalar, 0.0, 1.0)

    def __repr__(self) -> str:
        return f"MultiplicativeNoise(multiplier={self.multiplier}, p={self.p})"


class ISONoise(PhotometricTransform[NoiseParam]):
    r"""Camera-sensor-like noise (Albumentations ``ISONoise``, approximate).

    Adds luminance Gaussian noise scaled by ``intensity`` plus a small
    saturation perturbation from ``color_shift``.
    """

    def __init__(
        self,
        color_shift: tuple[float, float] = (0.01, 0.05),
        intensity: tuple[float, float] = (0.1, 0.5),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.color_shift = color_shift
        self.intensity = intensity

    def make_params(self, img: Tensor) -> NoiseParam:
        return NoiseParam(std=_random.uniform(self.intensity[0], self.intensity[1]))

    def _apply_image(self, img: Tensor, params: NoiseParam) -> Tensor:
        cs = _random.uniform(self.color_shift[0], self.color_shift[1])
        out = F.adjust_saturation(img, 1.0 + cs)
        noise = lucid.randn(*out.shape) * (params.std * 0.1)
        return lucid.clip(out + noise, 0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"ISONoise(color_shift={self.color_shift}, intensity={self.intensity}, "
            f"p={self.p})"
        )


class Downscale(PhotometricTransform[ScaleParam]):
    r"""Downscale then upscale to lose high-frequency detail (Albumentations ``Downscale``).

    Samples a scale factor uniformly from ``[scale_min, scale_max]``,
    resizes the image to that fraction of its original ``(H, W)`` via
    nearest-neighbour, then resizes back to the original size — a
    cheap way to simulate sensor / JPEG block-level degradation.

    Parameters
    ----------
    scale_min : float, optional, default=0.25
        Lower bound of the downscale factor.
    scale_max : float, optional, default=0.25
        Upper bound of the downscale factor (set equal to
        ``scale_min`` for a deterministic factor).
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(
        self,
        scale_min: float = 0.25,
        scale_max: float = 0.25,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.scale_min = scale_min
        self.scale_max = scale_max

    def make_params(self, img: Tensor) -> ScaleParam:
        return ScaleParam(scale=_random.uniform(self.scale_min, self.scale_max))

    def _apply_image(self, img: Tensor, params: ScaleParam) -> Tensor:
        h, w = F._spatial_hw(img)
        dh = max(int(round(h * params.scale)), 1)
        dw = max(int(round(w * params.scale)), 1)
        small = F.resize(img, (dh, dw), interpolation="nearest")
        return F.resize(small, (h, w), interpolation="nearest")

    def __repr__(self) -> str:
        return (
            f"Downscale(scale_min={self.scale_min}, scale_max={self.scale_max}, "
            f"p={self.p})"
        )


# ── B8: defocus / zoom blur ─────────────────────────────────────────


@dataclass(frozen=True)
class RadiusParam:
    radius: int


class Defocus(PhotometricTransform[RadiusParam]):
    r"""Disk-kernel (out-of-focus) blur (Albumentations ``Defocus``).

    Samples an integer radius ``r`` from ``radius`` (inclusive) and
    convolves with a normalised disk of radius ``r`` — pixels inside
    the disk get weight ``1 / area``, those outside get zero.
    Approximates the bokeh circle of an out-of-focus lens better than
    Gaussian blur.

    Parameters
    ----------
    radius : (int, int), optional, default=(3, 10)
        Inclusive sampling range for the disk radius in pixels.
        Larger radii produce more aggressive defocus.
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(self, radius: tuple[int, int] = (3, 10), p: float = 0.5) -> None:
        super().__init__(p=p)
        self.radius = radius

    def make_params(self, img: Tensor) -> RadiusParam:
        return RadiusParam(radius=_random.randint(self.radius[0], self.radius[1] + 1))

    def _apply_image(self, img: Tensor, params: RadiusParam) -> Tensor:
        r = params.radius
        k = 2 * r + 1
        disk = [
            [1.0 if (i - r) ** 2 + (j - r) ** 2 <= r * r else 0.0 for j in range(k)]
            for i in range(k)
        ]
        total = sum(sum(row) for row in disk) or 1.0
        disk = [[v / total for v in row] for row in disk]
        return F.depthwise_conv2d(img, disk)

    def __repr__(self) -> str:
        return f"Defocus(radius={self.radius}, p={self.p})"


@dataclass(frozen=True)
class ZoomParam:
    factor: float


class ZoomBlur(PhotometricTransform[ZoomParam]):
    r"""Radial zoom blur via averaged centre-zoomed copies (Albumentations ``ZoomBlur``).

    Samples a final zoom factor uniformly from ``[1.0, max_factor]``,
    builds 5 centre-crops at progressively-increasing zoom levels
    (each resized back to the input ``(H, W)`` via bilinear), and
    averages them with the original image — simulates a fast forward
    dolly into the scene.

    Parameters
    ----------
    max_factor : float, optional, default=1.31
        Upper bound on the final zoom factor (must be ``>= 1.0``).
    step_factor : (float, float), optional, default=(0.01, 0.03)
        Reserved for future per-step jitter; currently the 5 zoom
        levels are evenly spaced toward ``max_factor`` regardless.
    p : float, optional, default=0.5
        Probability of applying the transform.
    """

    def __init__(
        self,
        max_factor: float = 1.31,
        step_factor: tuple[float, float] = (0.01, 0.03),
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.max_factor = max_factor
        self.step_factor = step_factor

    def make_params(self, img: Tensor) -> ZoomParam:
        return ZoomParam(factor=_random.uniform(1.0, self.max_factor))

    def _apply_image(self, img: Tensor, params: ZoomParam) -> Tensor:
        h, w = F._spatial_hw(img)
        n = 5
        acc = img
        for i in range(1, n + 1):
            f = 1.0 + (params.factor - 1.0) * i / n
            ch, cw = max(int(round(h / f)), 1), max(int(round(w / f)), 1)
            top, left = (h - ch) // 2, (w - cw) // 2
            zoomed = F.resize(
                F.crop(img, top, left, ch, cw), (h, w), interpolation="bilinear"
            )
            acc = acc + zoomed
        return lucid.clip(acc / (n + 1), 0.0, 1.0)

    def __repr__(self) -> str:
        return f"ZoomBlur(max_factor={self.max_factor}, p={self.p})"
