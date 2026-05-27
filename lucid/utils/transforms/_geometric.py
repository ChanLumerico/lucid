"""Geometric transforms — spatial resampling / cropping / flipping.

Albumentations-compatible class names + constructor signatures, built on
Lucid's typed multi-target :class:`~lucid.utils.transforms._base.
GeometricTransform`: every transform here moves the image, its mask
(nearest resampling), bounding boxes, and keypoints consistently.
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
)
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    Keypoints,
    crop_boxes,
    crop_keypoints,
    flip_boxes,
    flip_keypoints,
    resize_boxes,
    resize_keypoints,
)
from lucid.utils.transforms._interpolation import Interpolation, as_interpolation


# ── parameter types ─────────────────────────────────────────────────


@dataclass(frozen=True)
class CropBox:
    """A crop window: top-left plus extent."""

    top: int
    left: int
    height: int
    width: int


# ── resize family ───────────────────────────────────────────────────


class Resize(_NoParams, GeometricTransform[Empty]):
    r"""Resize to an exact ``height`` x ``width`` (Albumentations ``Resize``).

    Parameters
    ----------
    height, width : int
        Target spatial size.
    interpolation : int or str or Interpolation, optional, default=1
        Image resampling mode (OpenCV codes accepted; masks use nearest).
    p : float, optional, default=1.0
    """

    def __init__(
        self,
        height: int,
        width: int,
        interpolation: int | str | Interpolation = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.interpolation = as_interpolation(interpolation)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.resize(img, (self.height, self.width), interpolation=self.interpolation)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return F.resize(mask, (self.height, self.width), interpolation=Interpolation.NEAREST)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        return resize_boxes(boxes, self.height, self.width)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        return resize_keypoints(kps, self.height, self.width)

    def __repr__(self) -> str:
        return f"Resize(height={self.height}, width={self.width}, p={self.p})"


class _MaxSizeResize(_NoParams, GeometricTransform[Empty]):
    """Shared base for SmallestMaxSize / LongestMaxSize."""

    def __init__(
        self,
        max_size: int,
        interpolation: int | str | Interpolation = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.max_size = max_size
        self.interpolation = as_interpolation(interpolation)

    def _target(self, h: int, w: int) -> tuple[int, int]:
        raise NotImplementedError

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        h, w = F._spatial_hw(img)
        return F.resize(img, self._target(h, w), interpolation=self.interpolation)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        h, w = F._spatial_hw(mask)
        return F.resize(mask, self._target(h, w), interpolation=Interpolation.NEAREST)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        new_h, new_w = self._target(*boxes.canvas_size)
        return resize_boxes(boxes, new_h, new_w)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        new_h, new_w = self._target(*kps.canvas_size)
        return resize_keypoints(kps, new_h, new_w)


class SmallestMaxSize(_MaxSizeResize):
    r"""Scale so the **shorter** side equals ``max_size`` (aspect kept)."""

    def _target(self, h: int, w: int) -> tuple[int, int]:
        scale = self.max_size / min(h, w)
        return int(round(h * scale)), int(round(w * scale))

    def __repr__(self) -> str:
        return f"SmallestMaxSize(max_size={self.max_size}, p={self.p})"


class LongestMaxSize(_MaxSizeResize):
    r"""Scale so the **longer** side equals ``max_size`` (aspect kept)."""

    def _target(self, h: int, w: int) -> tuple[int, int]:
        scale = self.max_size / max(h, w)
        return int(round(h * scale)), int(round(w * scale))

    def __repr__(self) -> str:
        return f"LongestMaxSize(max_size={self.max_size}, p={self.p})"


# ── crop family ─────────────────────────────────────────────────────


class CenterCrop(_NoParams, GeometricTransform[Empty]):
    r"""Crop a centered ``height`` x ``width`` window (Albumentations ``CenterCrop``)."""

    def __init__(self, height: int, width: int, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.height = height
        self.width = width

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.center_crop(img, (self.height, self.width))

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return F.center_crop(mask, (self.height, self.width))

    def _offsets(self, canvas: tuple[int, int]) -> tuple[int, int]:
        h, w = canvas
        return max((h - self.height) // 2, 0), max((w - self.width) // 2, 0)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        top, left = self._offsets(boxes.canvas_size)
        return crop_boxes(boxes, top, left, self.height, self.width)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        top, left = self._offsets(kps.canvas_size)
        return crop_keypoints(kps, top, left, self.height, self.width)

    def __repr__(self) -> str:
        return f"CenterCrop(height={self.height}, width={self.width}, p={self.p})"


@dataclass(frozen=True)
class Offset:
    top: int
    left: int


class RandomCrop(GeometricTransform[Offset]):
    r"""Crop a random ``height`` x ``width`` window (Albumentations ``RandomCrop``)."""

    def __init__(self, height: int, width: int, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.height = height
        self.width = width

    def make_params(self, img: Tensor) -> Offset:
        h, w = F._spatial_hw(img)
        return Offset(
            top=_random.randint(0, h - self.height + 1),
            left=_random.randint(0, w - self.width + 1),
        )

    def _apply_image(self, img: Tensor, params: Offset) -> Tensor:
        return F.crop(img, params.top, params.left, self.height, self.width)

    def _apply_mask(self, mask: Tensor, params: Offset) -> Tensor:
        return F.crop(mask, params.top, params.left, self.height, self.width)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Offset) -> BoundingBoxes:
        return crop_boxes(boxes, params.top, params.left, self.height, self.width)

    def _apply_keypoints(self, kps: Keypoints, params: Offset) -> Keypoints:
        return crop_keypoints(kps, params.top, params.left, self.height, self.width)

    def __repr__(self) -> str:
        return f"RandomCrop(height={self.height}, width={self.width}, p={self.p})"


class RandomResizedCrop(GeometricTransform[CropBox]):
    r"""Crop a random area/aspect region then resize to ``height`` x ``width``.

    Albumentations ``RandomResizedCrop``.
    """

    def __init__(
        self,
        height: int,
        width: int,
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: int | str | Interpolation = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.height = height
        self.width = width
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
            img, params.top, params.left, params.height, params.width,
            (self.height, self.width), interpolation=self.interpolation,
        )

    def _apply_mask(self, mask: Tensor, params: CropBox) -> Tensor:
        return F.resized_crop(
            mask, params.top, params.left, params.height, params.width,
            (self.height, self.width), interpolation=Interpolation.NEAREST,
        )

    def _apply_boxes(self, boxes: BoundingBoxes, params: CropBox) -> BoundingBoxes:
        cropped = crop_boxes(boxes, params.top, params.left, params.height, params.width)
        return resize_boxes(cropped, self.height, self.width)

    def _apply_keypoints(self, kps: Keypoints, params: CropBox) -> Keypoints:
        cropped = crop_keypoints(
            kps, params.top, params.left, params.height, params.width
        )
        return resize_keypoints(cropped, self.height, self.width)

    def __repr__(self) -> str:
        return (
            f"RandomResizedCrop(height={self.height}, width={self.width}, "
            f"scale={self.scale}, ratio={self.ratio}, p={self.p})"
        )


# ── flips ───────────────────────────────────────────────────────────


class HorizontalFlip(_NoParams, GeometricTransform[Empty]):
    r"""Flip left-right with probability ``p`` (Albumentations ``HorizontalFlip``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.hflip(img)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return F.hflip(mask)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        return flip_boxes(boxes, horizontal=True)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        return flip_keypoints(kps, horizontal=True)

    def __repr__(self) -> str:
        return f"HorizontalFlip(p={self.p})"


class VerticalFlip(_NoParams, GeometricTransform[Empty]):
    r"""Flip top-bottom with probability ``p`` (Albumentations ``VerticalFlip``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return F.vflip(img)

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return F.vflip(mask)

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        return flip_boxes(boxes, horizontal=False)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        return flip_keypoints(kps, horizontal=False)

    def __repr__(self) -> str:
        return f"VerticalFlip(p={self.p})"
