"""
lucid.nn.functional._utils — interpolate / rotate / one_hot.

All routes are 1:1 to fused C++ kernels in `_C_nn`. The legacy Python
compositions are gone; the only remaining Python helper is `area` mode
of `interpolate`, which is a thin wrapper over `avg_pool2d` (still a
single C++ op call, no numpy fallback).
"""

from __future__ import annotations

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of, to_engine_dtype
from lucid.types import Numeric, _Scalar


# --------------------------------------------------------------------------- #
# Interpolate (4 modes)
# --------------------------------------------------------------------------- #

def _interpolate_bilinear(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    H_out, W_out = size
    return Tensor._wrap(_C_nn.interpolate_bilinear(
        impl_of(input_), int(H_out), int(W_out), bool(align_corners)))


def _interpolate_trilinear(
    input_: Tensor, size: tuple[int, int, int], align_corners: bool = False
) -> Tensor:
    D_out, H_out, W_out = size
    return Tensor._wrap(_C_nn.interpolate_trilinear(
        impl_of(input_), int(D_out), int(H_out), int(W_out),
        bool(align_corners)))


def _interpolate_nearest(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    # align_corners has no effect for nearest in PyTorch semantics.
    H_out, W_out = size
    return Tensor._wrap(_C_nn.interpolate_nearest_2d(
        impl_of(input_), int(H_out), int(W_out)))


def _interpolate_nearest_3d(
    input_: Tensor, size: tuple[int, int, int], align_corners: bool = False
) -> Tensor:
    D_out, H_out, W_out = size
    return Tensor._wrap(_C_nn.interpolate_nearest_3d(
        impl_of(input_), int(D_out), int(H_out), int(W_out)))


def _interpolate_area(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    # area mode === avg_pool with stride=kernel=floor(in/out). PyTorch parity.
    from lucid.nn import functional as F
    _, _, H, W = input_.shape
    out_h, out_w = size
    kh = max(int(H // out_h), 1)
    kw = max(int(W // out_w), 1)
    pooled = F.avg_pool2d(input_, kernel_size=(kh, kw), stride=(kh, kw))
    return pooled[:, :, :out_h, :out_w]


# --------------------------------------------------------------------------- #
# Rotate
# --------------------------------------------------------------------------- #

def rotate(
    input_: Tensor, angle: float, center: tuple[_Scalar, _Scalar] | None = None
) -> Tensor:
    if input_.ndim != 4:
        raise ValueError("rotate: input must be 4-D (N, C, H, W)")
    _, _, H, W = input_.shape
    if center is None:
        cy, cx = H / 2.0, W / 2.0
    else:
        cx, cy = float(center[0]), float(center[1])
    return Tensor._wrap(_C_nn.rotate(
        impl_of(input_), float(angle), float(cy), float(cx)))


# --------------------------------------------------------------------------- #
# One-hot
# --------------------------------------------------------------------------- #

def one_hot(
    input_: Tensor, num_classes: int = -1, dtype: Numeric | bool | None = None
) -> Tensor:
    if input_.dtype.base_dtype is not int:
        raise TypeError("one_hot only supports integer input.")
    if num_classes == -1:
        # Best-effort: infer from data.  Materializes once on host.
        import lucid
        num_classes = int(lucid.max(input_).item()) + 1
    from lucid.types import Int8
    out_dtype = dtype if dtype is not None else Int8
    eng_dt = to_engine_dtype(out_dtype)
    return Tensor._wrap(_C_nn.one_hot(
        impl_of(input_), int(num_classes), eng_dt))
