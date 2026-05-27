"""Geometric transforms — spatial resampling / cropping / flipping.

Deterministic inference transforms (:class:`Resize`, :class:`CenterCrop`,
:class:`Pad`) and randomized augmentations (:class:`RandomCrop`,
:class:`RandomResizedCrop`, :class:`RandomHorizontalFlip`,
:class:`RandomVerticalFlip`).  Every transform here is a
:class:`~lucid.utils.transforms._base.GeometricTransform`, so it moves
masks (nearest resampling) and bounding boxes (coordinate transform)
consistently with the image.
"""

import math
from dataclasses import dataclass

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import (
    Empty,
    GeometricTransform,
    _NoParams,
    _ProbabilityGate,
)
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    crop_boxes,
    flip_boxes,
    pad_boxes,
    resize_boxes,
)
from lucid.utils.transforms._interpolation import Interpolation, as_interpolation


def _as_hw(size: int | tuple[int, int]) -> tuple[int, int]:
    """Normalize a size arg to ``(h, w)`` (square when an int)."""
    return (size, size) if isinstance(size, int) else (size[0], size[1])


# ── parameter types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class Offset:
    """Top-left crop offset (crop size is fixed on the transform)."""

    top: int
    left: int


@dataclass(frozen=True)
class CropBox:
    """A crop window: top-left plus extent."""

    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True)
class FlipParams:
    """Whether this call flips."""

    apply: bool


# ── deterministic ───────────────────────────────────────────────────


class Resize(_NoParams, GeometricTransform[Empty]):
    r"""Resize an image (and resample mask / scale boxes).

    Parameters
    ----------
    size : int or (int, int)
        ``int`` scales the shorter side (aspect preserved); ``(h, w)``
        resizes exactly.
    interpolation : str or Interpolation, optional, default="bilinear"
        Image resampling mode.  Masks always use nearest.
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.size = size
        self.interpolation = as_interpolation(interpolation)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.resize(img, self.size, interpolation=self.interpolation)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return F.resize(mask, self.size, interpolation=Interpolation.NEAREST)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = boxes.canvas_size
        new_h, new_w = F.resize_target(h, w, self.size)
        return resize_boxes(boxes, new_h, new_w)

    def __repr__(self) -> str:
        return f"Resize(size={self.size}, interpolation={self.interpolation})"


class CenterCrop(_NoParams, GeometricTransform[Empty]):
    r"""Crop a centered ``size`` window (square when an int)."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = size

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.center_crop(img, self.size)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return F.center_crop(mask, self.size)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = boxes.canvas_size
        th, tw = _as_hw(self.size)
        top = max((h - th) // 2, 0)
        left = max((w - tw) // 2, 0)
        return crop_boxes(boxes, top, left, th, tw)

    def __repr__(self) -> str:
        return f"CenterCrop(size={self.size})"


class Pad(_NoParams, GeometricTransform[Empty]):
    r"""Pad spatial borders by a fixed amount.

    Parameters
    ----------
    padding : int or (left, right, top, bottom)
        Border widths (a single int pads all four sides).
    fill : float, optional, default=0.0
        Constant fill value (``"constant"`` mode).
    mode : str, optional, default="constant"
        Padding mode.
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

    def _lrtb(self) -> tuple[int, int, int, int]:
        if isinstance(self.padding, int):
            return self.padding, self.padding, self.padding, self.padding
        return self.padding

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.pad(img, self.padding, mode=self.mode, value=self.fill)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return F.pad(mask, self.padding, mode=self.mode, value=self.fill)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        left, right, top, bottom = self._lrtb()
        h, w = boxes.canvas_size
        return pad_boxes(boxes, left, top, h + top + bottom, w + left + right)

    def __repr__(self) -> str:
        return f"Pad(padding={self.padding}, fill={self.fill}, mode={self.mode!r})"


# ── randomized ──────────────────────────────────────────────────────


class RandomHorizontalFlip(_ProbabilityGate, GeometricTransform[FlipParams]):
    r"""Horizontally flip with probability ``p`` (mirrors mask + boxes)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def make_params(self, img: Tensor) -> FlipParams:
        return FlipParams(apply=self._gate())

    def _apply_image(self, img: Tensor, params: FlipParams) -> Tensor:
        return F.hflip(img) if params.apply else img

    def _apply_mask(self, mask: Tensor, params: FlipParams) -> Tensor:
        return F.hflip(mask) if params.apply else mask

    def _apply_boxes(self, boxes: BoundingBoxes, params: FlipParams) -> BoundingBoxes:
        return flip_boxes(boxes, horizontal=True) if params.apply else boxes

    def __repr__(self) -> str:
        return f"RandomHorizontalFlip(p={self.p})"


class RandomVerticalFlip(_ProbabilityGate, GeometricTransform[FlipParams]):
    r"""Vertically flip with probability ``p`` (mirrors mask + boxes)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p)

    def make_params(self, img: Tensor) -> FlipParams:
        return FlipParams(apply=self._gate())

    def _apply_image(self, img: Tensor, params: FlipParams) -> Tensor:
        return F.vflip(img) if params.apply else img

    def _apply_mask(self, mask: Tensor, params: FlipParams) -> Tensor:
        return F.vflip(mask) if params.apply else mask

    def _apply_boxes(self, boxes: BoundingBoxes, params: FlipParams) -> BoundingBoxes:
        return flip_boxes(boxes, horizontal=False) if params.apply else boxes

    def __repr__(self) -> str:
        return f"RandomVerticalFlip(p={self.p})"


class RandomCrop(GeometricTransform[Offset]):
    r"""Crop a random ``size`` window (optionally padding first).

    Parameters
    ----------
    size : int or (int, int)
        Output crop size (square when an int).
    padding : int or (left, right, top, bottom), optional
        Pad before cropping (e.g. CIFAR ``RandomCrop(32, padding=4)``).
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

    def _padded_hw(self, h: int, w: int) -> tuple[int, int]:
        if self.padding is None:
            return h, w
        if isinstance(self.padding, int):
            return h + 2 * self.padding, w + 2 * self.padding
        left, right, top, bottom = self.padding
        return h + top + bottom, w + left + right

    def make_params(self, img: Tensor) -> Offset:
        h, w = self._padded_hw(*F._spatial_hw(img))
        th, tw = _as_hw(self.size)
        return Offset(
            top=_random.randint(0, h - th + 1),
            left=_random.randint(0, w - tw + 1),
        )

    def _crop_image_like(self, x: Tensor, params: Offset) -> Tensor:
        if self.padding is not None:
            x = F.pad(x, self.padding, mode=self.padding_mode, value=self.fill)
        th, tw = _as_hw(self.size)
        return F.crop(x, params.top, params.left, th, tw)

    def _apply_image(self, img: Tensor, params: Offset) -> Tensor:
        return self._crop_image_like(img, params)

    def _apply_mask(self, mask: Tensor, params: Offset) -> Tensor:
        return self._crop_image_like(mask, params)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Offset) -> BoundingBoxes:
        if self.padding is not None:
            if isinstance(self.padding, int):
                left = top = right = bottom = self.padding
            else:
                left, right, top, bottom = self.padding
            h, w = boxes.canvas_size
            boxes = pad_boxes(boxes, left, top, h + top + bottom, w + left + right)
        th, tw = _as_hw(self.size)
        return crop_boxes(boxes, params.top, params.left, th, tw)

    def __repr__(self) -> str:
        return f"RandomCrop(size={self.size}, padding={self.padding})"


class RandomResizedCrop(GeometricTransform[CropBox]):
    r"""Crop a random area/aspect region, then resize to ``size``.

    The canonical ImageNet training augmentation (Inception-style).

    Parameters
    ----------
    size : int or (int, int)
        Output size; an ``int`` means a **square** output.
    scale : (float, float), optional, default=(0.08, 1.0)
        Range of cropped-area fraction.
    ratio : (float, float), optional, default=(3/4, 4/3)
        Range of aspect ratios (width / height).
    interpolation : str or Interpolation, optional, default="bilinear"
        Image resampling mode (masks use nearest).
    """

    def __init__(
        self,
        size: int | tuple[int, int],
        *,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str | Interpolation = Interpolation.BILINEAR,
    ) -> None:
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = as_interpolation(interpolation)

    def make_params(self, img: Tensor) -> CropBox:
        h, w = F._spatial_hw(img)
        area = float(h * w)
        log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
        for _ in range(10):
            target_area = _random.uniform(self.scale[0], self.scale[1]) * area
            aspect = math.exp(_random.uniform(log_ratio[0], log_ratio[1]))
            cw = int(round(math.sqrt(target_area * aspect)))
            ch = int(round(math.sqrt(target_area / aspect)))
            if 0 < cw <= w and 0 < ch <= h:
                return CropBox(
                    top=_random.randint(0, h - ch + 1),
                    left=_random.randint(0, w - cw + 1),
                    height=ch,
                    width=cw,
                )
        # Fallback: largest center region fitting the aspect range.
        in_ratio = w / h
        if in_ratio < self.ratio[0]:
            cw, ch = w, int(round(w / self.ratio[0]))
        elif in_ratio > self.ratio[1]:
            ch, cw = h, int(round(h * self.ratio[1]))
        else:
            cw, ch = w, h
        return CropBox(top=(h - ch) // 2, left=(w - cw) // 2, height=ch, width=cw)

    def _apply_image(self, img: Tensor, params: CropBox) -> Tensor:
        return F.resized_crop(
            img,
            params.top,
            params.left,
            params.height,
            params.width,
            _as_hw(self.size),
            interpolation=self.interpolation,
        )

    def _apply_mask(self, mask: Tensor, params: CropBox) -> Tensor:
        return F.resized_crop(
            mask,
            params.top,
            params.left,
            params.height,
            params.width,
            _as_hw(self.size),
            interpolation=Interpolation.NEAREST,
        )

    def _apply_boxes(self, boxes: BoundingBoxes, params: CropBox) -> BoundingBoxes:
        cropped = crop_boxes(
            boxes, params.top, params.left, params.height, params.width
        )
        out_h, out_w = _as_hw(self.size)
        return resize_boxes(cropped, out_h, out_w)

    def __repr__(self) -> str:
        return (
            f"RandomResizedCrop(size={self.size}, scale={self.scale}, "
            f"ratio={self.ratio})"
        )
