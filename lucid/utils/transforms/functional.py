"""Stateless functional transforms.

Mirrors ``torchvision.transforms.functional``: plain functions that take
a :class:`lucid.Tensor` image (``(C, H, W)`` or ``(B, C, H, W)``) and
return a transformed tensor.  The class-based transforms in
:mod:`lucid.utils.transforms` are thin wrappers over these.

All operations are Lucid-native (engine ops only) — no numpy / PIL.
"""

import lucid
import lucid.nn.functional as F
from lucid._tensor import Tensor

_RESIZE_ALIGN_MODES = frozenset({"bilinear", "bicubic", "linear", "trilinear"})


def _spatial_hw(img: Tensor) -> tuple[int, int]:
    """Return ``(H, W)`` for a CHW or NCHW tensor."""
    if img.ndim not in (3, 4):
        raise ValueError(
            f"expected a (C, H, W) or (B, C, H, W) tensor, got ndim={img.ndim}"
        )
    return int(img.shape[-2]), int(img.shape[-1])


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

    if isinstance(size, int):
        if h <= w:
            new_h, new_w = size, int(round(w * size / h))
        else:
            new_h, new_w = int(round(h * size / w)), size
    else:
        new_h, new_w = int(size[0]), int(size[1])

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
