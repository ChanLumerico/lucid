"""Stateless functional transforms.

Mirrors ``torchvision.transforms.functional``: plain functions that take
a :class:`lucid.Tensor` image (``(C, H, W)`` or ``(B, C, H, W)``) and
return a transformed tensor.  The class-based transforms in
:mod:`lucid.utils.transforms` are thin wrappers over these.

All operations are Lucid-native (engine ops only) — no numpy / PIL.
"""

import math
from typing import cast

import lucid
import lucid.nn.functional as F
from lucid._tensor import Tensor

_RESIZE_ALIGN_MODES = frozenset({"bilinear", "bicubic", "linear", "trilinear"})


def _cat(tensors: list[Tensor], dim: int) -> Tensor:
    """``lucid.concat`` wrapper (absorbs the int32-typed ``dim`` stub)."""
    return lucid.concat(tensors, dim=dim)  # type: ignore[arg-type]


def _inv(x: Tensor) -> Tensor:
    """``lucid.linalg.inv`` wrapper with a precise return type."""
    return cast(Tensor, lucid.linalg.inv(x))


def _spatial_hw(img: Tensor) -> tuple[int, int]:
    """Return ``(H, W)`` for a CHW or NCHW tensor."""
    if img.ndim not in (3, 4):
        raise ValueError(
            f"expected a (C, H, W) or (B, C, H, W) tensor, got ndim={img.ndim}"
        )
    return int(img.shape[-2]), int(img.shape[-1])


def resize_target(h: int, w: int, size: int | tuple[int, int]) -> tuple[int, int]:
    """Compute :func:`resize`'s output ``(H, W)`` for input ``(h, w)``.

    Shared by :func:`resize` and the mask / bounding-box paths so they
    stay consistent (shorter-side rule for an int ``size``).
    """
    if isinstance(size, int):
        if h <= w:
            return size, int(round(w * size / h))
        return int(round(h * size / w)), size
    return int(size[0]), int(size[1])


def resize(
    img: Tensor,
    size: int | tuple[int, int],
    *,
    interpolation: str = "bilinear",
) -> Tensor:
    r"""Resize an image.

    Parameters
    ----------
    img : Tensor
        Image tensor ``(C, H, W)`` or ``(B, C, H, W)``.
    size : int or (int, int)
        If an ``int``, the **shorter** side is scaled to ``size`` with
        the aspect ratio preserved (torchvision ``Resize(int)``
        semantics).  If ``(h, w)``, the image is resized to exactly that.
    interpolation : str, optional, default="bilinear"
        Mode forwarded to :func:`lucid.nn.functional.interpolate`.

    Returns
    -------
    Tensor
        Resized image with the same rank as the input.
    """
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    h, w = _spatial_hw(x)
    new_h, new_w = resize_target(h, w, size)

    align = False if interpolation in _RESIZE_ALIGN_MODES else None
    x = F.interpolate(x, size=(new_h, new_w), mode=interpolation, align_corners=align)
    return x[0] if unbatched else x


def crop(img: Tensor, top: int, left: int, height: int, width: int) -> Tensor:
    """Crop a ``height`` × ``width`` window with top-left at ``(top, left)``."""
    return img[..., top : top + height, left : left + width]


def center_crop(img: Tensor, size: int | tuple[int, int]) -> Tensor:
    r"""Crop a centered window of ``size`` (square if ``size`` is an int)."""
    crop_h, crop_w = (size, size) if isinstance(size, int) else (size[0], size[1])
    h, w = _spatial_hw(img)
    top = max((h - crop_h) // 2, 0)
    left = max((w - crop_w) // 2, 0)
    return crop(img, top, left, crop_h, crop_w)


def hflip(img: Tensor) -> Tensor:
    """Horizontally flip (mirror along the width axis)."""
    # lucid.flip's stub types dims as int32; a plain int works at runtime.
    return lucid.flip(img, dims=-1)  # type: ignore[arg-type]


def vflip(img: Tensor) -> Tensor:
    """Vertically flip (mirror along the height axis)."""
    return lucid.flip(img, dims=-2)  # type: ignore[arg-type]


def pad(
    img: Tensor,
    padding: int | tuple[int, int, int, int],
    *,
    mode: str = "constant",
    value: float = 0.0,
) -> Tensor:
    r"""Pad spatial borders.

    Parameters
    ----------
    img : Tensor
        Image ``(C, H, W)`` or ``(B, C, H, W)``.
    padding : int or (left, right, top, bottom)
        Border widths.  A single int pads all four sides equally.
    mode : str, optional, default="constant"
        Forwarded to :func:`lucid.nn.functional.pad`.
    value : float, optional, default=0.0
        Fill value for ``"constant"`` mode.
    """
    if isinstance(padding, int):
        pad_spec = (padding, padding, padding, padding)
    else:
        pad_spec = padding
    return F.pad(img, pad_spec, mode=mode, value=value)


def normalize(
    img: Tensor,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> Tensor:
    r"""Normalize per channel: ``(img - mean) / std``.

    ``mean`` / ``std`` are broadcast over the channel axis.
    """
    c = int(img.shape[-3])
    mean_t = lucid.tensor(list(mean), dtype=img.dtype).reshape(1, c, 1, 1)
    std_t = lucid.tensor(list(std), dtype=img.dtype).reshape(1, c, 1, 1)
    if img.ndim == 3:
        mean_t = mean_t[0]
        std_t = std_t[0]
    return (img - mean_t) / std_t


def rescale(img: Tensor, scale: float = 1.0 / 255.0) -> Tensor:
    """Multiply pixel values by ``scale`` (e.g. uint8 ``[0,255]`` → ``[0,1]``)."""
    return img * scale


def resized_crop(
    img: Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    size: int | tuple[int, int],
    *,
    interpolation: str = "bilinear",
) -> Tensor:
    """Crop ``(top, left, height, width)`` then resize to ``size``."""
    return resize(
        crop(img, top, left, height, width), size, interpolation=interpolation
    )


def rgb_to_grayscale(img: Tensor, *, keep_channels: bool = True) -> Tensor:
    r"""Convert an RGB image to luminance (ITU-R 601-2 weights).

    Parameters
    ----------
    img : Tensor
        RGB image with 3 channels.
    keep_channels : bool, optional, default=True
        If ``True`` the single luma channel is broadcast back to 3
        channels (handy for ``saturation`` blending); else a 1-channel
        result is returned.
    """
    c = int(img.shape[-3])
    if c != 3:
        raise ValueError(f"rgb_to_grayscale expects 3 channels, got {c}")
    r = img[..., 0:1, :, :]
    g = img[..., 1:2, :, :]
    b = img[..., 2:3, :, :]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    if keep_channels:
        return _cat([gray, gray, gray], -3)
    return gray


def _blend(img1: Tensor, img2: Tensor, ratio: float) -> Tensor:
    """``ratio * img1 + (1 - ratio) * img2``, clipped to ``[0, 1]``."""
    return lucid.clip(ratio * img1 + (1.0 - ratio) * img2, 0.0, 1.0)


def adjust_brightness(img: Tensor, factor: float) -> Tensor:
    """Scale brightness (blend toward black). ``factor=1`` is a no-op."""
    return _blend(img, img * 0.0, factor)


def adjust_contrast(img: Tensor, factor: float) -> Tensor:
    """Adjust contrast (blend toward the per-image mean gray)."""
    gray = rgb_to_grayscale(img, keep_channels=False)
    mean = lucid.mean(gray, dim=[-1, -2, -3], keepdim=True)
    return _blend(img, mean, factor)


def adjust_saturation(img: Tensor, factor: float) -> Tensor:
    """Adjust saturation (blend toward grayscale). ``factor=1`` is a no-op."""
    return _blend(img, rgb_to_grayscale(img, keep_channels=True), factor)


def adjust_hue(img: Tensor, factor: float) -> Tensor:
    r"""Shift hue by ``factor`` (in ``[-0.5, 0.5]``) via an HSV round-trip.

    ``factor`` is added to the hue channel (wrapped mod 1.0); ``0`` is a
    no-op.  Implemented with Lucid tensor ops only (no colour-space
    library).
    """
    if factor == 0.0:
        return img
    r = img[..., 0:1, :, :]
    g = img[..., 1:2, :, :]
    b = img[..., 2:3, :, :]

    maxc = lucid.max(img, dim=-3, keepdim=True)
    minc = lucid.min(img, dim=-3, keepdim=True)
    delta = maxc - minc
    eps = 1e-10

    # Hue (in [0, 1)), following the standard piecewise definition.
    rc = (maxc - r) / (delta + eps)
    gc = (maxc - g) / (delta + eps)
    bc = (maxc - b) / (delta + eps)
    h_r = bc - gc
    h_g = 2.0 + rc - bc
    h_b = 4.0 + gc - rc
    hue = lucid.where(maxc == r, h_r, lucid.where(maxc == g, h_g, h_b))

    def _frac1(z: Tensor) -> Tensor:
        # z mod 1.0 via z - floor(z) (Tensor has no % operator).
        return z - lucid.floor(z)

    hue = _frac1(hue / 6.0)
    # Achromatic pixels (delta == 0) have undefined hue → set 0.
    hue = lucid.where(delta < eps, hue * 0.0, hue)

    hue = _frac1(hue + factor)

    # HSV → RGB with S, V recovered from max/min.
    s = delta / (maxc + eps)
    v = maxc
    i = lucid.floor(hue * 6.0)
    f = hue * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i_mod = i - 6.0 * lucid.floor(i / 6.0)

    def _sel(
        c0: Tensor, c1: Tensor, c2: Tensor, c3: Tensor, c4: Tensor, c5: Tensor
    ) -> Tensor:
        out = lucid.where(i_mod == 0, c0, c1)
        out = lucid.where(i_mod == 2, c2, out)
        out = lucid.where(i_mod == 3, c3, out)
        out = lucid.where(i_mod == 4, c4, out)
        out = lucid.where(i_mod == 5, c5, out)
        out = lucid.where(i_mod == 1, c1, out)
        return out

    new_r = _sel(v, q, p, p, t, v)
    new_g = _sel(t, v, v, q, p, p)
    new_b = _sel(p, p, t, v, v, q)
    rgb = _cat([new_r, new_g, new_b], -3)
    # Achromatic pixels stay unchanged.  lucid.where needs a same-shape
    # condition, so broadcast the per-pixel delta mask across channels.
    achromatic = _cat([delta < eps, delta < eps, delta < eps], -3)
    return lucid.clip(lucid.where(achromatic, img, rgb), 0.0, 1.0)


# ── affine / perspective warps ──────────────────────────────────────
#
# Coordinate convention matches OpenCV (so Albumentations-style matrices
# transfer directly): a forward 3x3 matrix ``M`` maps an *input* pixel
# ``[x, y, 1]`` to an *output* pixel.  Images are sampled by inverting M
# into ``affine_grid``'s normalized output→input ``theta``; boxes and
# keypoints apply M to their coordinates directly.


def _norm_to_pixel(h: int, w: int) -> Tensor:
    """3x3 matrix mapping ``align_corners=True`` normalized coords → pixels.

    With ``align_corners=True`` the corner pixels map to ``-1`` / ``+1``:
    ``px = (nx + 1) * (W - 1) / 2``.  (Lucid's ``align_corners=False``
    grid path does not reproduce an identity ``theta``, so the warp path
    standardises on ``align_corners=True``.)
    """
    return lucid.tensor(
        [
            [(w - 1) / 2.0, 0.0, (w - 1) / 2.0],
            [0.0, (h - 1) / 2.0, (h - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )


def affine_theta(
    matrix: Tensor, in_hw: tuple[int, int], out_hw: tuple[int, int]
) -> Tensor:
    """Convert a forward pixel matrix ``M`` to ``affine_grid`` ``theta`` (1,2,3)."""
    ih, iw = in_hw
    oh, ow = out_hw
    no = _norm_to_pixel(oh, ow)
    ni_inv = _inv(_norm_to_pixel(ih, iw))
    m_inv = _inv(matrix)
    theta = lucid.matmul(lucid.matmul(ni_inv, m_inv), no)  # norm_out → norm_in
    return theta[:2].reshape(1, 2, 3)


def warp_affine(
    img: Tensor,
    matrix: Tensor,
    out_hw: tuple[int, int],
    *,
    mode: str = "bilinear",
    fill: float = 0.0,
) -> Tensor:
    """Warp ``img`` by forward pixel matrix ``matrix`` (affine or homography)."""
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    b, c, ih, iw = (int(d) for d in x.shape)
    theta = affine_theta(matrix, (ih, iw), out_hw)
    if b > 1:
        theta = _cat([theta] * b, 0)
    grid = F.affine_grid(theta, [b, c, out_hw[0], out_hw[1]], align_corners=True)
    gmode = "nearest" if mode == "nearest" else "bilinear"
    pad = "zeros" if fill == 0.0 else "border"
    y = F.grid_sample(x, grid, mode=gmode, padding_mode=pad, align_corners=True)
    return y[0] if unbatched else y


def affine_points(pts: Tensor, matrix: Tensor) -> Tensor:
    """Apply a forward 3x3 ``matrix`` to ``(N, 2)`` ``(x, y)`` points."""
    n = int(pts.shape[0])
    ones = lucid.ones(n, 1, dtype=pts.dtype)
    hom = _cat([pts, ones], 1)  # (N, 3)
    out = lucid.matmul(hom, lucid.swapaxes(matrix, 0, 1))  # type: ignore[arg-type]  # (N, 3)
    return out[:, :2] / out[:, 2:3]  # perspective divide (no-op for affine)


def rotation_matrix(
    angle_deg: float, cx: float, cy: float, scale: float = 1.0
) -> Tensor:
    """OpenCV ``getRotationMatrix2D`` forward matrix (CCW degrees)."""
    rad = math.radians(angle_deg)
    alpha = scale * math.cos(rad)
    beta = scale * math.sin(rad)
    return lucid.tensor(
        [
            [alpha, beta, (1.0 - alpha) * cx - beta * cy],
            [-beta, alpha, beta * cx + (1.0 - alpha) * cy],
            [0.0, 0.0, 1.0],
        ]
    )


def affine_matrix(
    *,
    cx: float,
    cy: float,
    scale: float = 1.0,
    angle_deg: float = 0.0,
    shear_x_deg: float = 0.0,
    shear_y_deg: float = 0.0,
    translate_x: float = 0.0,
    translate_y: float = 0.0,
) -> Tensor:
    """Compose a forward affine matrix about ``(cx, cy)`` then translate."""
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    shx, shy = math.tan(math.radians(shear_x_deg)), math.tan(math.radians(shear_y_deg))
    # Rotate ∘ shear ∘ scale (about origin).
    a = scale * (cos_a - sin_a * shy)
    bb = scale * (cos_a * shx - sin_a)
    cc = scale * (sin_a + cos_a * shy)
    dd = scale * (sin_a * shx + cos_a)
    # Translate so the center maps to itself, then add the requested shift.
    e = cx - a * cx - bb * cy + translate_x
    f = cy - cc * cx - dd * cy + translate_y
    return lucid.tensor([[a, bb, e], [cc, dd, f], [0.0, 0.0, 1.0]])


def perspective_matrix(src: list[list[float]], dst: list[list[float]]) -> Tensor:
    """Forward homography mapping the 4 ``src`` corners to ``dst`` (xy each).

    Solves the 8-DoF system for the 3x3 matrix (``h33 = 1``).
    """
    rows: list[list[float]] = []
    rhs: list[float] = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        rows.append([sx, sy, 1.0, 0.0, 0.0, 0.0, -dx * sx, -dx * sy])
        rhs.append(dx)
        rows.append([0.0, 0.0, 0.0, sx, sy, 1.0, -dy * sx, -dy * sy])
        rhs.append(dy)
    a: Tensor = lucid.tensor(rows)
    b: Tensor = lucid.tensor([[v] for v in rhs])
    h = lucid.matmul(_inv(a), b).reshape(-1)
    hl = h.numpy().tolist()
    return lucid.tensor(
        [[hl[0], hl[1], hl[2]], [hl[3], hl[4], hl[5]], [hl[6], hl[7], 1.0]]
    )


# ── blur + displacement-field warps ─────────────────────────────────


def gaussian_blur(img: Tensor, sigma: float, ksize: int | None = None) -> Tensor:
    """Separable Gaussian blur (depthwise conv). ``sigma`` in pixels."""
    if ksize is None:
        ksize = int(2 * round(3.0 * sigma) + 1)
    if ksize % 2 == 0:
        ksize += 1
    half = ksize // 2
    w1 = [math.exp(-((i - half) ** 2) / (2.0 * sigma * sigma)) for i in range(ksize)]
    s = sum(w1)
    w1 = [v / s for v in w1]

    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    c = int(x.shape[1])
    kx = _cat([lucid.tensor(w1).reshape(1, 1, 1, ksize)] * c, 0)
    ky = _cat([lucid.tensor(w1).reshape(1, 1, ksize, 1)] * c, 0)
    x = F.conv2d(x, kx, padding=(0, half), groups=c)
    x = F.conv2d(x, ky, padding=(half, 0), groups=c)
    return x[0] if unbatched else x


def _pixel_grid(h: int, w: int) -> tuple[Tensor, Tensor]:
    """Return ``(yy, xx)`` float pixel-coordinate grids of shape ``(H, W)``."""
    yy, xx = lucid.meshgrid(
        lucid.arange(0, h, dtype=lucid.float32),
        lucid.arange(0, w, dtype=lucid.float32),
        indexing="ij",
    )
    return yy, xx


def remap(img: Tensor, dx: Tensor, dy: Tensor, *, mode: str = "bilinear") -> Tensor:
    """Backward warp: ``out(y, x) = img(x + dx, y + dy)`` via ``grid_sample``."""
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    b, _, h, w = (int(d) for d in x.shape)
    yy, xx = _pixel_grid(h, w)
    gx = 2.0 * (xx + dx) / (w - 1) - 1.0
    gy = 2.0 * (yy + dy) / (h - 1) - 1.0
    grid = lucid.stack([gx, gy], dim=-1)[None]  # (1, H, W, 2)
    if b > 1:
        grid = _cat([grid] * b, 0)
    gmode = "nearest" if mode == "nearest" else "bilinear"
    y = F.grid_sample(
        x, grid, mode=gmode, padding_mode="reflection", align_corners=True
    )
    return y[0] if unbatched else y


def sample_field_at_points(
    field: Tensor, pts: Tensor, canvas_hw: tuple[int, int]
) -> Tensor:
    """Bilinearly sample a ``(H, W)`` field at ``(N, 2)`` ``(x, y)`` points → ``(N, 1)``."""
    h, w = canvas_hw
    n = int(pts.shape[0])
    gx = 2.0 * pts[:, 0:1] / (w - 1) - 1.0
    gy = 2.0 * pts[:, 1:2] / (h - 1) - 1.0
    grid = _cat([gx, gy], 1).reshape(1, n, 1, 2)
    samp = F.grid_sample(
        field.reshape(1, 1, h, w), grid, mode="bilinear", align_corners=True
    )
    return samp.reshape(n, 1)


# ── histogram + kernel helpers (colour batch) ───────────────────────


def _equalize_channel(ch: Tensor, clip_limit: float | None = None) -> Tensor:
    """Histogram-equalize a single ``(H, W)`` channel in ``[0, 1]``."""
    h, w = int(ch.shape[0]), int(ch.shape[1])
    idx = lucid.clip(lucid.round(ch * 255.0), 0.0, 255.0).long().reshape(-1)
    hist = lucid.bincount(idx, minlength=256).to(lucid.float32)  # type: ignore[arg-type]
    if clip_limit is not None:
        cap = clip_limit * (h * w) / 256.0
        clipped = lucid.clip(hist, 0.0, cap)
        excess = (hist - clipped).sum() / 256.0
        hist = clipped + excess
    cdf = lucid.cumsum(hist)
    big = lucid.ones(256, dtype=lucid.float32) * float(h * w + 1)
    cdf_min = lucid.min(lucid.where(cdf > 0.0, cdf, big))
    total = cdf[255]
    denom = float(total.item()) - float(cdf_min.item())
    if denom <= 0.0:  # flat channel — leave unchanged
        return ch
    lut = (cdf - cdf_min) / denom * 255.0
    return lucid.take(lut, idx).reshape(h, w) / 255.0


def equalize(img: Tensor, clip_limit: float | None = None) -> Tensor:
    """Per-channel histogram equalization of a ``[0, 1]`` image."""
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    b, c, h, w = (int(d) for d in x.shape)
    imgs = []
    for bi in range(b):
        chans = [
            _equalize_channel(x[bi, ci], clip_limit).reshape(1, h, w) for ci in range(c)
        ]
        imgs.append(_cat(chans, 0).reshape(1, c, h, w))
    out = _cat(imgs, 0)
    return out[0] if unbatched else out


def depthwise_conv2d(img: Tensor, kernel2d: list[list[float]]) -> Tensor:
    """Apply a 2-D kernel to every channel independently (same padding)."""
    kh, kw = len(kernel2d), len(kernel2d[0])
    unbatched = img.ndim == 3
    x = img[None] if unbatched else img
    c = int(x.shape[1])
    k = lucid.tensor(kernel2d).reshape(1, 1, kh, kw)
    weight = _cat([k] * c, 0)
    y = F.conv2d(x, weight, padding=(kh // 2, kw // 2), groups=c)
    return y[0] if unbatched else y
