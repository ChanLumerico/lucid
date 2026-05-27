"""Geometric transforms — spatial resampling / cropping.

Deterministic inference transforms (:class:`Resize`, :class:`CenterCrop`)
plus randomized training augmentations (:class:`RandomCrop`,
:class:`RandomResizedCrop`, :class:`RandomHorizontalFlip`,
:class:`RandomVerticalFlip`, :class:`Pad`).  Randomized transforms sample
their parameters once in :meth:`make_params` (honouring
:func:`lucid.manual_seed`).
"""

import math
from typing import cast

from lucid._tensor import Tensor
from lucid.utils.transforms import functional as F
from lucid.utils.transforms import _random
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


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    return (size, size) if isinstance(size, int) else (size[0], size[1])


class Pad(Transform):
    r"""Pad spatial borders by a fixed amount.

    Parameters
    ----------
    padding : int or (left, right, top, bottom)
        Border widths (a single int pads all four sides).
    fill : float, optional, default=0.0
        Constant fill value (``"constant"`` mode).
    mode : str, optional, default="constant"
        Padding mode (see :func:`lucid.nn.functional.pad`).
    """

    def __init__(
        self,
        padding: int | tuple[int, int, int, int],
        *,
        fill: float = 0.0,
        mode: str = "constant",
    ) -> None:
        self.padding = padding
        self.fill = fill
        self.mode = mode

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return F.pad(img, self.padding, mode=self.mode, value=self.fill)

    def __repr__(self) -> str:
        return f"Pad(padding={self.padding}, fill={self.fill}, mode={self.mode!r})"


class RandomHorizontalFlip(Transform):
    r"""Horizontally flip with probability ``p``.

    Parameters
    ----------
    p : float, optional, default=0.5
        Flip probability.
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def make_params(self, img: Tensor) -> dict[str, object]:
        return {"flip": _random.rand() < self.p}

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return F.hflip(img) if params["flip"] else img

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip(Transform):
    r"""Vertically flip with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def make_params(self, img: Tensor) -> dict[str, object]:
        return {"flip": _random.rand() < self.p}

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        return F.vflip(img) if params["flip"] else img

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomCrop(Transform):
    r"""Crop a random ``size`` window (optionally padding first).

    Parameters
    ----------
    size : int or (int, int)
        Output crop size (square if an int).
    padding : int or (left, right, top, bottom), optional
        If given, the image is padded before cropping (common for CIFAR
        ``RandomCrop(32, padding=4)``).
    fill : float, optional, default=0.0
        Pad fill value.
    padding_mode : str, optional, default="constant"
        Pad mode.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        padding: int | tuple[int, int, int, int] | None = None,
        fill: float = 0.0,
        padding_mode: str = "constant",
    ) -> None:
        self.size = size
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def _padded_hw(self, img: Tensor) -> tuple[int, int]:
        h, w = F._spatial_hw(img)
        if self.padding is None:
            return h, w
        if isinstance(self.padding, int):
            return h + 2 * self.padding, w + 2 * self.padding
        left, right, top, bottom = self.padding
        return h + top + bottom, w + left + right

    def make_params(self, img: Tensor) -> dict[str, object]:
        h, w = self._padded_hw(img)
        th, tw = _as_hw(self.size)
        top = _random.randint(0, h - th + 1)
        left = _random.randint(0, w - tw + 1)
        return {"top": top, "left": left}

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        if self.padding is not None:
            img = F.pad(img, self.padding, mode=self.padding_mode, value=self.fill)
        th, tw = _as_hw(self.size)
        top = cast(int, params["top"])
        left = cast(int, params["left"])
        return F.crop(img, top, left, th, tw)

    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size}, padding={self.padding})"


class RandomResizedCrop(Transform):
    r"""Crop a random area/aspect region, then resize to ``size``.

    The canonical ImageNet training augmentation (Inception-style): a
    region covering ``scale`` fraction of the image area with aspect
    ratio in ``ratio`` is cropped and resized to ``size``.

    Parameters
    ----------
    size : int or (int, int)
        Output size after resizing.
    scale : (float, float), optional, default=(0.08, 1.0)
        Range of area fraction to crop.
    ratio : (float, float), optional, default=(3/4, 4/3)
        Range of aspect ratios (width / height) to crop.
    interpolation : str, optional, default="bilinear"
        Resize mode.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "bilinear",
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def make_params(self, img: Tensor) -> dict[str, object]:
        h, w = F._spatial_hw(img)
        area = float(h * w)
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = _random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(_random.uniform(log_ratio[0], log_ratio[1]))
            cw = int(round(math.sqrt(target_area * aspect)))
            ch = int(round(math.sqrt(target_area / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                top = _random.randint(0, h - ch + 1)
                left = _random.randint(0, w - cw + 1)
                return {"top": top, "left": left, "height": ch, "width": cw}
        # Fallback: center crop the largest aspect-fitting region.
        in_ratio = w / h
        if in_ratio < self.ratio[0]:
            cw, ch = w, int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            ch, cw = h, int(round(h * self.ratio[1]))
        else:
            cw, ch = w, h
        top = (h - ch) // 2
        left = (w - cw) // 2
        return {"top": top, "left": left, "height": ch, "width": cw}

    def _apply_image(self, img: Tensor, params: dict[str, object]) -> Tensor:
        # RandomResizedCrop's int ``size`` means a *square* output
        # (unlike Resize(int), which scales the shorter side).
        out_size = _as_hw(self.size)
        return F.resized_crop(
            img,
            cast(int, params["top"]),
            cast(int, params["left"]),
            cast(int, params["height"]),
            cast(int, params["width"]),
            out_size,
            interpolation=self.interpolation,
        )

    def __repr__(self) -> str:
        return (
            f"RandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio={self.ratio})"
        )
