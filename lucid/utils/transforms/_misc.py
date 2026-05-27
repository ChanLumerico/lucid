"""Cross-target + utility transforms.

Transforms whose parameters depend on a *companion* target (boxes /
mask), or that wrap user callables.  These override :meth:`__call__`
because the standard ``make_params(image)`` hook only sees the image.

* ``Lambda`` — per-target user functions.
* ``MaskDropout`` — drop random mask label regions (image + mask).
* ``BBoxSafeRandomCrop`` / ``RandomSizedBBoxSafeCrop`` /
  ``RandomCropNearBBox`` — detection-aware crops (need the bounding
  boxes to choose the crop window).
"""

from typing import Callable

import lucid
from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, Transform, _NoParams, _find_reference
from lucid.utils.transforms._crop import Crop
from lucid.utils.transforms._datatypes import BoundingBoxes, Image, Mask, to_xyxy
from lucid.utils.transforms._geometric import Resize


class Lambda(_NoParams, Transform[Empty]):
    r"""Apply user callables per target type (Albumentations ``Lambda``).

    Parameters
    ----------
    image, mask : callable, optional
        ``Tensor -> Tensor`` applied to image / mask tensors.
    bboxes, keypoints : callable, optional
        Applied to the :class:`BoundingBoxes` / :class:`Keypoints` object.
    name : str, optional
    p : float, optional, default=1.0
    """

    def __init__(
        self,
        image: Callable[[Tensor], Tensor] | None = None,
        mask: Callable[[Tensor], Tensor] | None = None,
        bboxes: Callable[[BoundingBoxes], BoundingBoxes] | None = None,
        keypoints: Callable[[object], object] | None = None,
        name: str | None = None,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.image_fn = image
        self.mask_fn = mask
        self.bboxes_fn = bboxes
        self.keypoints_fn = keypoints
        self.name = name

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return self.image_fn(img) if self.image_fn is not None else img

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return self.mask_fn(mask) if self.mask_fn is not None else mask

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        return self.bboxes_fn(boxes) if self.bboxes_fn is not None else boxes

    def __repr__(self) -> str:
        return f"Lambda(name={self.name!r}, p={self.p})"


def _union_xyxy(boxes: BoundingBoxes) -> tuple[float, float, float, float]:
    xy = to_xyxy(boxes)
    x1 = float(lucid.min(xy[:, 0]).item())
    y1 = float(lucid.min(xy[:, 1]).item())
    x2 = float(lucid.max(xy[:, 2]).item())
    y2 = float(lucid.max(xy[:, 3]).item())
    return x1, y1, x2, y2


class _BBoxAwareCrop:
    """Mixin: find boxes in a sample + compute a box-containing crop window."""

    erosion_rate: float

    def _find_boxes(self, inputs: object) -> BoundingBoxes | None:
        if isinstance(inputs, BoundingBoxes):
            return inputs
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, BoundingBoxes):
                    return v
        return None

    def _safe_window(self, boxes: BoundingBoxes) -> tuple[int, int, int, int]:
        h, w = boxes.canvas_size
        x1, y1, x2, y2 = _union_xyxy(boxes)
        # Erode the must-contain region toward its center.
        ex = (x2 - x1) * self.erosion_rate / 2.0
        ey = (y2 - y1) * self.erosion_rate / 2.0
        x1, y1, x2, y2 = x1 + ex, y1 + ey, x2 - ex, y2 - ey
        cx_min = _random.randint(0, int(max(x1, 0)) + 1)
        cy_min = _random.randint(0, int(max(y1, 0)) + 1)
        cx_max = _random.randint(int(min(x2, w)), w + 1)
        cy_max = _random.randint(int(min(y2, h)), h + 1)
        cx_max = max(cx_max, cx_min + 1)
        cy_max = max(cy_max, cy_min + 1)
        return cx_min, cy_min, cx_max, cy_max


class BBoxSafeRandomCrop(_NoParams, Transform[Empty], _BBoxAwareCrop):
    r"""Random crop guaranteed to contain every bounding box (Albu ``BBoxSafeRandomCrop``)."""

    def __init__(self, erosion_rate: float = 0.0, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.erosion_rate = erosion_rate

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return img  # real work happens in __call__

    def __call__(self, inputs: object) -> object:
        ref = _find_reference(inputs)
        if ref is None:
            raise ValueError("BBoxSafeRandomCrop: no image in the sample")
        if self.p < 1.0 and _random.rand() >= self.p:
            return inputs
        boxes = self._find_boxes(inputs)
        if boxes is None:
            h, w = F._spatial_hw(ref)
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            x0, y0, x1, y1 = self._safe_window(boxes)
        return Crop(x0, y0, x1, y1, p=1.0)(inputs)

    def __repr__(self) -> str:
        return f"BBoxSafeRandomCrop(erosion_rate={self.erosion_rate}, p={self.p})"


class RandomSizedBBoxSafeCrop(_NoParams, Transform[Empty], _BBoxAwareCrop):
    r"""Box-safe crop then resize to ``height`` x ``width`` (Albu ``RandomSizedBBoxSafeCrop``)."""

    def __init__(
        self,
        height: int,
        width: int,
        erosion_rate: float = 0.0,
        interpolation: int | str = 1,
        p: float = 1.0,
    ) -> None:
        super().__init__(p=p)
        self.height = height
        self.width = width
        self.erosion_rate = erosion_rate
        self.interpolation = interpolation

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return img

    def __call__(self, inputs: object) -> object:
        ref = _find_reference(inputs)
        if ref is None:
            raise ValueError("RandomSizedBBoxSafeCrop: no image in the sample")
        if self.p < 1.0 and _random.rand() >= self.p:
            return inputs
        boxes = self._find_boxes(inputs)
        if boxes is None:
            h, w = F._spatial_hw(ref)
            window = (0, 0, w, h)
        else:
            window = self._safe_window(boxes)
        cropped = Crop(*window, p=1.0)(inputs)
        return Resize(self.height, self.width, interpolation=self.interpolation, p=1.0)(
            cropped
        )

    def __repr__(self) -> str:
        return (
            f"RandomSizedBBoxSafeCrop(height={self.height}, width={self.width}, "
            f"erosion_rate={self.erosion_rate}, p={self.p})"
        )


class RandomCropNearBBox(_NoParams, Transform[Empty]):
    r"""Crop a window around the first bounding box (Albumentations ``RandomCropNearBBox``).

    Parameters
    ----------
    max_part_shift : float, optional, default=0.3
        Max jitter of the crop window edges, as a fraction of the box size.
    p : float, optional, default=1.0
    """

    def __init__(self, max_part_shift: float = 0.3, p: float = 1.0) -> None:
        super().__init__(p=p)
        self.max_part_shift = max_part_shift

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return img

    def __call__(self, inputs: object) -> object:
        ref = _find_reference(inputs)
        if ref is None:
            raise ValueError("RandomCropNearBBox: no image in the sample")
        if self.p < 1.0 and _random.rand() >= self.p:
            return inputs
        boxes = None
        if isinstance(inputs, dict):
            for v in inputs.values():
                if isinstance(v, BoundingBoxes):
                    boxes = v
                    break
        h, w = F._spatial_hw(ref)
        if boxes is None:
            return inputs
        x1, y1, x2, y2 = _union_xyxy(boxes)
        bw, bh = x2 - x1, y2 - y1
        jx, jy = bw * self.max_part_shift, bh * self.max_part_shift
        cx0 = max(int(x1 - _random.uniform(0, jx)), 0)
        cy0 = max(int(y1 - _random.uniform(0, jy)), 0)
        cx1 = min(int(x2 + _random.uniform(0, jx)), w)
        cy1 = min(int(y2 + _random.uniform(0, jy)), h)
        return Crop(cx0, cy0, max(cx1, cx0 + 1), max(cy1, cy0 + 1), p=1.0)(inputs)

    def __repr__(self) -> str:
        return f"RandomCropNearBBox(max_part_shift={self.max_part_shift}, p={self.p})"


class MaskDropout(_NoParams, Transform[Empty]):
    r"""Drop random mask label regions from image + mask (Albumentations ``MaskDropout``).

    Parameters
    ----------
    max_objects : int, optional, default=1
        Max number of distinct mask labels to zero out.
    image_fill_value : float, optional, default=0.0
    mask_fill_value : float, optional, default=0.0
    p : float, optional, default=0.5
    """

    def __init__(
        self,
        max_objects: int = 1,
        image_fill_value: float = 0.0,
        mask_fill_value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.max_objects = max_objects
        self.image_fill_value = image_fill_value
        self.mask_fill_value = mask_fill_value

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return img

    def __call__(self, inputs: object) -> object:
        if not isinstance(inputs, dict):
            return inputs
        if self.p < 1.0 and _random.rand() >= self.p:
            return inputs
        mask_obj = next((v for v in inputs.values() if isinstance(v, Mask)), None)
        img_obj = next((v for v in inputs.values() if isinstance(v, Image)), None)
        if mask_obj is None:
            return inputs

        labels = sorted(
            {int(round(v)) for v in mask_obj.data.reshape(-1).numpy().tolist()}
        )
        labels = [v for v in labels if v != 0]
        if not labels:
            return inputs
        n = _random.randint(1, self.max_objects + 1)
        chosen: list[int] = []
        pool = list(labels)
        for _ in range(min(n, len(pool))):
            k = _random.randint(0, len(pool))
            chosen.append(pool.pop(k))

        keep = lucid.ones(*mask_obj.data.shape, dtype=mask_obj.data.dtype)
        for lab in chosen:
            keep = keep * (mask_obj.data != lab).to(mask_obj.data.dtype)

        out = dict(inputs)
        new_mask = mask_obj.data * keep + self.mask_fill_value * (1.0 - keep)
        for key, v in inputs.items():
            if v is mask_obj:
                out[key] = Mask(new_mask)
            elif v is img_obj and img_obj is not None:
                c = int(img_obj.data.shape[-3])
                kc = F._cat([keep] * c, 0) if keep.ndim == 3 else keep
                out[key] = Image(img_obj.data * kc + self.image_fill_value * (1.0 - kc))
        return out

    def __repr__(self) -> str:
        return f"MaskDropout(max_objects={self.max_objects}, p={self.p})"
