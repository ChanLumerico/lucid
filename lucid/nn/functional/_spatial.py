"""
lucid.nn.functional._spatial — affine_grid + grid_sample.

Both routes are 1:1 to fused C++ kernels in `_C_nn`. The C++ ops do the
full forward + backward (analytic gradient through bilinear interpolation)
without any Python composition.
"""

from __future__ import annotations

from typing import Literal

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of


_PaddingType = Literal["zeros", "border"]
_InterpolateType = Literal["bilinear", "nearest"]


_MODE_CODES = {"bilinear": 0, "nearest": 1}
_PAD_CODES = {"zeros": 0, "border": 1}


def affine_grid(
    theta: Tensor, size: tuple[int, ...], align_corners: bool = True
) -> Tensor:
    if len(size) != 4:
        raise ValueError("affine_grid: size must be (N, C, H, W) for 2-D grids")
    N, _, H, W = size
    return Tensor._wrap(_C_nn.affine_grid(
        impl_of(theta), int(N), int(H), int(W), bool(align_corners)))


def grid_sample(
    input_: Tensor,
    grid: Tensor,
    mode: _InterpolateType = "bilinear",
    padding_mode: _PaddingType = "zeros",
    align_corners: bool = True,
) -> Tensor:
    if mode not in _MODE_CODES:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
    if padding_mode not in _PAD_CODES:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")
    return Tensor._wrap(_C_nn.grid_sample(
        impl_of(input_), impl_of(grid),
        _MODE_CODES[mode], _PAD_CODES[padding_mode], bool(align_corners)))
