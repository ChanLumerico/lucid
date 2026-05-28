"""Crop / pad family — Albumentations-compatible.

``Crop``, ``PadIfNeeded``, ``RandomSizedCrop``, ``CropAndPad`` — all
geometric, moving mask / boxes / keypoints with the image.
"""

from dataclasses import dataclass

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, GeometricTransform, _NoParams
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    Keypoints,
    crop_boxes,
    crop_keypoints,
    pad_boxes,
    pad_keypoints,
    resize_boxes,
    resize_keypoints,
)
from lucid.utils.transforms._interpolation import Interpolation, as_interpolation


@dataclass(frozen=True)
class CropBox:
    top: int
    left: int
    height: int
    width: int


class Crop(_NoParams, GeometricTransform[Empty]):
    r"""Crop a fixed region ``[x_min, y_min, x_max, y_max)`` (Albumentations ``Crop``).

    Deterministic — every call returns the same rectangle regardless
    of input size.  Masks, boxes, and keypoints are translated in
    lock-step with the image (``GeometricTransform`` contract).

    Parameters
    ----------
    x_min, y_min : int, optional, default=0
        Top-left corner of the crop window (inclusive).
    x_max, y_max : int, optional, default=1024
        Bottom-right corner (exclusive).
    p : float, optional, default=1.0
        Probability of applying the crop.  Below ``1.0``, the input
        passes through unchanged on ``1 - p`` of calls.
    """

    def __init__(
        self,
        x_min: int = 0,
        y_min: int = 0,
        x_max: int = 1024,
        y_max: int = 1024,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.x_min, self.y_min, self.x_max, self.y_max = x_min, y_min, x_max, y_max

    @property
    def _hw(self) -> tuple[int, int]:
        return self.y_max - self.y_min, self.x_max - self.x_min

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        h, w = self._hw
        return F.crop(img, self.y_min, self.x_min, h, w)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        h, w = self._hw
        return F.crop(mask, self.y_min, self.x_min, h, w)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = self._hw
        return crop_boxes(boxes, self.y_min, self.x_min, h, w)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        h, w = self._hw
        return crop_keypoints(kps, self.y_min, self.x_min, h, w)

    def __repr__(self) -> str:
        return (
            f"Crop(x_min={self.x_min}, y_min={self.y_min}, "
            f"x_max={self.x_max}, y_max={self.y_max}, p={self.p})"
        )


class PadIfNeeded(_NoParams, GeometricTransform[Empty]):
    r"""Pad to at least ``min_height`` x ``min_width`` (Albumentations ``PadIfNeeded``).

    Parameters
    ----------
    min_height, min_width : int
        Minimum output size; smaller inputs are centered and padded.
    border_mode : int, optional, default=4
    value, mask_value : float, optional, default=0.0
    p : float, optional, default=1.0
    """

    def __init__(
        self,
        min_height: int = 1024,
        min_width: int = 1024,
        border_mode: int = 4,
        value: float = 0.0,
        mask_value: float = 0.0,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.min_height = min_height
        self.min_width = min_width
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def _pads(self, h: int, w: int) -> tuple[int, int, int, int]:
        dh = max(self.min_height - h, 0)
        dw = max(self.min_width - w, 0)
        top, left = dh // 2, dw // 2
        return left, dw - left, top, dh - top  # (l, r, t, b)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        h, w = F._spatial_hw(img)
        return F.pad(img, self._pads(h, w), value=self.value)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        h, w = F._spatial_hw(mask)
        return F.pad(mask, self._pads(h, w), value=self.mask_value)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = boxes.canvas_size
        left, right, top, bottom = self._pads(h, w)
        return pad_boxes(boxes, left, top, h + top + bottom, w + left + right)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        h, w = kps.canvas_size
        left, right, top, bottom = self._pads(h, w)
        return pad_keypoints(kps, left, top, h + top + bottom, w + left + right)

    def __repr__(self) -> str:
        return (
            f"PadIfNeeded(min_height={self.min_height}, "
            f"min_width={self.min_width}, p={self.p})"
        )


class RandomSizedCrop(GeometricTransform[CropBox]):
    r"""Random-size crop then resize (Albumentations ``RandomSizedCrop``).

    Parameters
    ----------
    min_max_height : (int, int)
        Range of crop heights (pixels).
    height, width : int
        Output size after resizing.
    w2h_ratio : float, optional, default=1.0
        Crop width / height ratio.
    interpolation : int or str or Interpolation, optional, default=1
    p : float, optional, default=1.0
    """

    def __init__(
        self,
        min_max_height: tuple[int, int],
        height: int,
        width: int,
        w2h_ratio: float = 1.0,
        interpolation: int | str | Interpolation = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.min_max_height = min_max_height
        self.height = height
        self.width = width
        self.w2h_ratio = w2h_ratio
        self.interpolation = as_interpolation(interpolation)

    def make_params(self, img: Tensor) -> CropBox:
        h, w = F._spatial_hw(img)
        ch = _random.randint(self.min_max_height[0], self.min_max_height[1] + 1)
        cw = int(round(ch * self.w2h_ratio))
        ch, cw = min(ch, h), min(cw, w)
        return CropBox(
            top=_random.randint(0, h - ch + 1),
            left=_random.randint(0, w - cw + 1),
            height=ch,
            width=cw,
        )

    def _apply_image(self, img: Tensor, params: CropBox) -> Tensor:
        return F.resized_crop(
            img,
            params.top,
            params.left,
            params.height,
            params.width,
            (self.height, self.width),
            interpolation=self.interpolation,
        )

    def _apply_mask(self, mask: Tensor, params: CropBox) -> Tensor:
        return F.resized_crop(
            mask,
            params.top,
            params.left,
            params.height,
            params.width,
            (self.height, self.width),
            interpolation=Interpolation.NEAREST,
        )

    def _apply_boxes(self, boxes: BoundingBoxes, params: CropBox) -> BoundingBoxes:
        cropped = crop_boxes(
            boxes, params.top, params.left, params.height, params.width
        )
        return resize_boxes(cropped, self.height, self.width)

    def _apply_keypoints(self, kps: Keypoints, params: CropBox) -> Keypoints:
        cropped = crop_keypoints(
            kps, params.top, params.left, params.height, params.width
        )
        return resize_keypoints(cropped, self.height, self.width)

    def __repr__(self) -> str:
        return (
            f"RandomSizedCrop(min_max_height={self.min_max_height}, "
            f"height={self.height}, width={self.width}, p={self.p})"
        )


class CropAndPad(_NoParams, GeometricTransform[Empty]):
    r"""Crop (negative) or pad (positive) every side by ``px`` pixels.

    A simplified Albumentations ``CropAndPad`` supporting a single int
    ``px`` applied to all four sides.

    Parameters
    ----------
    px : int
        Positive pads, negative crops, on every side.
    value : float, optional, default=0.0
        Pad fill (for positive ``px``).
    p : float, optional, default=1.0
    """

    def __init__(self, px: int = 0, value: float = 0.0, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.px = px
        self.value = value

    def _img_like(self, x: Tensor) -> Tensor:
        if self.px >= 0:
            return F.pad(x, self.px, value=self.value)
        c = -self.px
        h, w = F._spatial_hw(x)
        return F.crop(x, c, c, h - 2 * c, w - 2 * c)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return self._img_like(img)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return self._img_like(mask)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        h, w = boxes.canvas_size
        if self.px >= 0:
            return pad_boxes(boxes, self.px, self.px, h + 2 * self.px, w + 2 * self.px)
        c = -self.px
        return crop_boxes(boxes, c, c, h - 2 * c, w - 2 * c)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        h, w = kps.canvas_size
        if self.px >= 0:
            return pad_keypoints(
                kps, self.px, self.px, h + 2 * self.px, w + 2 * self.px
            )
        c = -self.px
        return crop_keypoints(kps, c, c, h - 2 * c, w - 2 * c)

    def __repr__(self) -> str:
        return f"CropAndPad(px={self.px}, p={self.p})"
