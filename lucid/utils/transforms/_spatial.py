"""Spatial augmentations — flips/transpose/rotations and affine warps.

Albumentations-compatible: ``Transpose``, ``Flip``, ``RandomRotate90``,
``Rotate``, ``ShiftScaleRotate``, ``Affine``, ``Perspective``.  The warp
family shares one matrix → ``grid_sample`` path for images/masks and an
affine-points path for boxes/keypoints, so all targets stay aligned.
"""

from dataclasses import dataclass

from lucid._tensor import Tensor
from lucid.utils.transforms import _random
from lucid.utils.transforms import functional as F
from lucid.utils.transforms._base import Empty, GeometricTransform, _NoParams
from lucid.utils.transforms._datatypes import (
    BoundingBoxes,
    Keypoints,
    affine_boxes,
    affine_keypoints,
    rot90_boxes,
    rot90_keypoints,
    transpose_boxes,
    transpose_keypoints,
)
from lucid.utils.transforms._interpolation import Interpolation, as_interpolation

import lucid


def _to_range(value: float | tuple[float, float]) -> tuple[float, float]:
    """``v`` → ``(-v, v)``; a 2-tuple is returned unchanged."""
    if isinstance(value, (int, float)):
        return (-float(value), float(value))
    return (float(value[0]), float(value[1]))


# OpenCV border_mode code → grid_sample padding mode.
_BORDER_TO_PAD = {0: "zeros", 1: "border", 2: "reflection", 4: "reflection"}


def _pad_mode(border_mode: int) -> str:
    return _BORDER_TO_PAD.get(border_mode, "reflection")


# ── parameter types ─────────────────────────────────────────────────


@dataclass
class WarpParams:
    """A sampled forward pixel matrix + output size for one call."""

    matrix: Tensor
    out_hw: tuple[int, int]


@dataclass(frozen=True)
class FlipAxis:
    code: int  # 1 = horizontal, 0 = vertical, -1 = both


@dataclass(frozen=True)
class Rot90Param:
    k: int


# ── exact (no-interpolation) transforms ─────────────────────────────


class Transpose(_NoParams, GeometricTransform[Empty]):
    r"""Swap the H and W axes with probability ``p`` (Albumentations ``Transpose``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def _apply_image(self, img: Tensor, params: Empty) -> Tensor:
        return lucid.swapaxes(img, -1, -2)  # type: ignore[arg-type]

    def _apply_mask(self, mask: Tensor, params: Empty) -> Tensor:
        return lucid.swapaxes(mask, -1, -2)  # type: ignore[arg-type]

    def _apply_boxes(self, boxes: BoundingBoxes, params: Empty) -> BoundingBoxes:
        return transpose_boxes(boxes)

    def _apply_keypoints(self, kps: Keypoints, params: Empty) -> Keypoints:
        return transpose_keypoints(kps)

    def __repr__(self) -> str:
        return f"Transpose(p={self.p})"


class Flip(GeometricTransform[FlipAxis]):
    r"""Flip around a random axis (Albumentations ``Flip``).

    Samples ``code`` ∈ {0 (vertical), 1 (horizontal), -1 (both)}.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def make_params(self, img: Tensor) -> FlipAxis:
        return FlipAxis(code=_random.randint(0, 3) - 1)  # {-1, 0, 1}

    def _flip(self, x: Tensor, code: int) -> Tensor:
        if code >= 0:  # horizontal (code 1) component
            x = F.hflip(x)
        if code <= 0:  # vertical (code 0 / -1) component
            x = F.vflip(x)
        return x

    def _apply_image(self, img: Tensor, params: FlipAxis) -> Tensor:
        return self._flip(img, params.code)

    def _apply_mask(self, mask: Tensor, params: FlipAxis) -> Tensor:
        return self._flip(mask, params.code)

    def _apply_boxes(self, boxes: BoundingBoxes, params: FlipAxis) -> BoundingBoxes:
        from lucid.utils.transforms._datatypes import flip_boxes

        if params.code >= 0:
            boxes = flip_boxes(boxes, horizontal=True)
        if params.code <= 0:
            boxes = flip_boxes(boxes, horizontal=False)
        return boxes

    def _apply_keypoints(self, kps: Keypoints, params: FlipAxis) -> Keypoints:
        from lucid.utils.transforms._datatypes import flip_keypoints

        if params.code >= 0:
            kps = flip_keypoints(kps, horizontal=True)
        if params.code <= 0:
            kps = flip_keypoints(kps, horizontal=False)
        return kps

    def __repr__(self) -> str:
        return f"Flip(p={self.p})"


class RandomRotate90(GeometricTransform[Rot90Param]):
    r"""Rotate by a random multiple of 90° (Albumentations ``RandomRotate90``)."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__(p=p)

    def make_params(self, img: Tensor) -> Rot90Param:
        return Rot90Param(k=_random.randint(0, 4))  # 0..3

    def _apply_image(self, img: Tensor, params: Rot90Param) -> Tensor:
        return lucid.rot90(img, params.k, dims=(-2, -1))  # type: ignore[arg-type]

    def _apply_mask(self, mask: Tensor, params: Rot90Param) -> Tensor:
        return lucid.rot90(mask, params.k, dims=(-2, -1))  # type: ignore[arg-type]

    def _apply_boxes(self, boxes: BoundingBoxes, params: Rot90Param) -> BoundingBoxes:
        return rot90_boxes(boxes, params.k)

    def _apply_keypoints(self, kps: Keypoints, params: Rot90Param) -> Keypoints:
        return rot90_keypoints(kps, params.k)

    def __repr__(self) -> str:
        return f"RandomRotate90(p={self.p})"


# ── affine-warp family ──────────────────────────────────────────────


class _WarpTransform(GeometricTransform[WarpParams]):
    """Shared image/mask/box/keypoint application for matrix warps."""

    interpolation: Interpolation
    border_mode: int

    def _img_mode(self) -> str:
        return "nearest" if self.interpolation == Interpolation.NEAREST else "bilinear"

    def _apply_image(self, img: Tensor, params: WarpParams) -> Tensor:
        return F.warp_affine(
            img, params.matrix, params.out_hw, mode=self._img_mode(),
            fill=0.0 if _pad_mode(self.border_mode) == "zeros" else 0.0,
        )

    def _apply_mask(self, mask: Tensor, params: WarpParams) -> Tensor:
        return F.warp_affine(mask, params.matrix, params.out_hw, mode="nearest")

    def _apply_boxes(self, boxes: BoundingBoxes, params: WarpParams) -> BoundingBoxes:
        return affine_boxes(boxes, params.matrix, params.out_hw)

    def _apply_keypoints(self, kps: Keypoints, params: WarpParams) -> Keypoints:
        return affine_keypoints(kps, params.matrix, params.out_hw)


class Rotate(_WarpTransform):
    r"""Rotate by a random angle in ``[-limit, limit]`` (Albumentations ``Rotate``).

    Parameters
    ----------
    limit : float or (float, float), optional, default=90
        Rotation-angle range in degrees.
    interpolation : int or str or Interpolation, optional, default=1
    border_mode : int, optional, default=4
        OpenCV border code (0 constant, 1 replicate, 2/4 reflect).
    value : float, optional, default=0.0
        Constant fill (only used with ``border_mode=0``).
    p : float, optional, default=0.5
    """

    def __init__(
        self,
        limit: float | tuple[float, float] = 90,
        interpolation: int | str | Interpolation = 1,
        border_mode: int = 4,
        value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.limit = _to_range(limit)
        self.interpolation = as_interpolation(interpolation)
        self.border_mode = border_mode
        self.value = value

    def make_params(self, img: Tensor) -> WarpParams:
        h, w = F._spatial_hw(img)
        angle = _random.uniform(self.limit[0], self.limit[1])
        matrix = F.rotation_matrix(angle, (w - 1) / 2.0, (h - 1) / 2.0)
        return WarpParams(matrix=matrix, out_hw=(h, w))

    def __repr__(self) -> str:
        return f"Rotate(limit={self.limit}, p={self.p})"


class ShiftScaleRotate(_WarpTransform):
    r"""Random shift + scale + rotation (Albumentations ``ShiftScaleRotate``).

    Parameters
    ----------
    shift_limit : float or (float, float), optional, default=0.0625
        Fraction-of-size translation range (applied to both axes).
    scale_limit : float or (float, float), optional, default=0.1
        Scale-delta range (effective scale is ``1 + s``).
    rotate_limit : float or (float, float), optional, default=45
        Rotation-angle range in degrees.
    interpolation, border_mode, value, p
        As in :class:`Rotate`.
    """

    def __init__(
        self,
        shift_limit: float | tuple[float, float] = 0.0625,
        scale_limit: float | tuple[float, float] = 0.1,
        rotate_limit: float | tuple[float, float] = 45,
        interpolation: int | str | Interpolation = 1,
        border_mode: int = 4,
        value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.shift_limit = _to_range(shift_limit)
        self.scale_limit = _to_range(scale_limit)
        self.rotate_limit = _to_range(rotate_limit)
        self.interpolation = as_interpolation(interpolation)
        self.border_mode = border_mode
        self.value = value

    def make_params(self, img: Tensor) -> WarpParams:
        h, w = F._spatial_hw(img)
        angle = _random.uniform(self.rotate_limit[0], self.rotate_limit[1])
        scale = 1.0 + _random.uniform(self.scale_limit[0], self.scale_limit[1])
        dx = _random.uniform(self.shift_limit[0], self.shift_limit[1]) * w
        dy = _random.uniform(self.shift_limit[0], self.shift_limit[1]) * h
        matrix = F.affine_matrix(
            cx=(w - 1) / 2.0, cy=(h - 1) / 2.0, scale=scale, angle_deg=angle,
            translate_x=dx, translate_y=dy,
        )
        return WarpParams(matrix=matrix, out_hw=(h, w))

    def __repr__(self) -> str:
        return (
            f"ShiftScaleRotate(shift_limit={self.shift_limit}, "
            f"scale_limit={self.scale_limit}, rotate_limit={self.rotate_limit}, "
            f"p={self.p})"
        )


class Affine(_WarpTransform):
    r"""General affine: scale / translate / rotate / shear (Albumentations ``Affine``).

    Parameters
    ----------
    scale : float or (float, float), optional, default=1.0
        Uniform scale (sampled if a range).
    translate_percent : float or (float, float), optional
        Fraction-of-size translation (sampled per axis).
    rotate : float or (float, float), optional, default=0
        Rotation-angle range in degrees.
    shear : float or (float, float), optional, default=0
        Shear-angle range in degrees (applied to x).
    interpolation, border_mode, value, p
        As in :class:`Rotate`.
    """

    def __init__(
        self,
        scale: float | tuple[float, float] = 1.0,
        translate_percent: float | tuple[float, float] = 0.0,
        rotate: float | tuple[float, float] = 0.0,
        shear: float | tuple[float, float] = 0.0,
        interpolation: int | str | Interpolation = 1,
        border_mode: int = 4,
        value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.scale = scale if isinstance(scale, tuple) else (scale, scale)
        self.translate_percent = _to_range(translate_percent)
        self.rotate = _to_range(rotate)
        self.shear = _to_range(shear)
        self.interpolation = as_interpolation(interpolation)
        self.border_mode = border_mode
        self.value = value

    def make_params(self, img: Tensor) -> WarpParams:
        h, w = F._spatial_hw(img)
        s = _random.uniform(self.scale[0], self.scale[1])
        angle = _random.uniform(self.rotate[0], self.rotate[1])
        shear = _random.uniform(self.shear[0], self.shear[1])
        dx = _random.uniform(self.translate_percent[0], self.translate_percent[1]) * w
        dy = _random.uniform(self.translate_percent[0], self.translate_percent[1]) * h
        matrix = F.affine_matrix(
            cx=(w - 1) / 2.0, cy=(h - 1) / 2.0, scale=s, angle_deg=angle,
            shear_x_deg=shear, translate_x=dx, translate_y=dy,
        )
        return WarpParams(matrix=matrix, out_hw=(h, w))

    def __repr__(self) -> str:
        return (
            f"Affine(scale={self.scale}, translate_percent={self.translate_percent}, "
            f"rotate={self.rotate}, shear={self.shear}, p={self.p})"
        )


class Perspective(_WarpTransform):
    r"""Random four-point perspective warp (Albumentations ``Perspective``).

    Parameters
    ----------
    scale : float or (float, float), optional, default=(0.05, 0.1)
        Std (as a fraction of size) of the random corner displacement.
    interpolation, border_mode, value, p
        As in :class:`Rotate`.  ``keep_size`` is implicit (output keeps
        the input size).
    """

    def __init__(
        self,
        scale: float | tuple[float, float] = (0.05, 0.1),
        interpolation: int | str | Interpolation = 1,
        border_mode: int = 4,
        value: float = 0.0,
        p: float = 0.5,
    ) -> None:
        super().__init__(p=p)
        self.scale = scale if isinstance(scale, tuple) else (0.0, scale)
        self.interpolation = as_interpolation(interpolation)
        self.border_mode = border_mode
        self.value = value

    def make_params(self, img: Tensor) -> WarpParams:
        h, w = F._spatial_hw(img)
        frac = _random.uniform(self.scale[0], self.scale[1])
        src = [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]]
        dst = [
            [_random.uniform(-frac, frac) * w + sx,
             _random.uniform(-frac, frac) * h + sy]
            for sx, sy in src
        ]
        # Homography mapping the perturbed corners back to the canvas.
        matrix = F.perspective_matrix(dst, src)
        return WarpParams(matrix=matrix, out_hw=(h, w))

    def __repr__(self) -> str:
        return f"Perspective(scale={self.scale}, p={self.p})"
