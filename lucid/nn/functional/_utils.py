"""
lucid.nn.functional._utils — interpolation, rotate, one_hot.

Pure Python composition over engine ops; no engine kernels needed.
"""

from __future__ import annotations

import lucid

from lucid._tensor import Tensor
from lucid.types import _Scalar, Numeric


# --------------------------------------------------------------------------- #
# Interpolation
# --------------------------------------------------------------------------- #

def _interpolate_bilinear(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    _, _, H, W = input_.shape
    out_h, out_w = size
    if (H, W) == (out_h, out_w):
        return input_
    device = input_.device

    if align_corners:
        if out_h == 1:
            indices_h = lucid.zeros((out_h,), dtype=lucid.Float32, device=device)
        else:
            indices_h = (lucid.arange(out_h, dtype=lucid.Float32, device=device)
                         * (H - 1) / (out_h - 1))
        if out_w == 1:
            indices_w = lucid.zeros((out_w,), dtype=lucid.Float32, device=device)
        else:
            indices_w = (lucid.arange(out_w, dtype=lucid.Float32, device=device)
                         * (W - 1) / (out_w - 1))
    else:
        scale_h = H / out_h
        scale_w = W / out_w
        indices_h = (lucid.arange(out_h, dtype=lucid.Float32, device=device)
                     + 0.5) * scale_h - 0.5
        indices_w = (lucid.arange(out_w, dtype=lucid.Float32, device=device)
                     + 0.5) * scale_w - 0.5

    indices_h = indices_h.clip(0, H - 1)
    indices_w = indices_w.clip(0, W - 1)

    top_indices_f = lucid.floor(indices_h)
    left_indices_f = lucid.floor(indices_w)
    top_indices = top_indices_f.astype(lucid.Int)
    bot_indices = (top_indices_f + 1).clip(0, H - 1).astype(lucid.Int)
    left_indices = left_indices_f.astype(lucid.Int)
    right_indices = (left_indices_f + 1).clip(0, W - 1).astype(lucid.Int)

    h_lerp = indices_h - top_indices_f
    w_lerp = indices_w - left_indices_f

    top_left = input_[:, :, top_indices[:, None], left_indices]
    top_right = input_[:, :, top_indices[:, None], right_indices]
    bot_left = input_[:, :, bot_indices[:, None], left_indices]
    bot_right = input_[:, :, bot_indices[:, None], right_indices]

    top = top_left * (1 - w_lerp) + top_right * w_lerp
    bot = bot_left * (1 - w_lerp) + bot_right * w_lerp
    return top * (1 - h_lerp[:, None]) + bot * h_lerp[:, None]


def _interpolate_trilinear(
    input_: Tensor, size: tuple[int, int, int], align_corners: bool = False
) -> Tensor:
    _, _, D, H, W = input_.shape
    out_d, out_h, out_w = size
    if (D, H, W) == (out_d, out_h, out_w):
        return input_
    device = input_.device

    def _idx(out_n, in_n, ac):
        if ac:
            if out_n == 1:
                return lucid.zeros((out_n,), dtype=lucid.Float32, device=device)
            return (lucid.arange(out_n, dtype=lucid.Float32, device=device)
                    * (in_n - 1) / (out_n - 1))
        return (lucid.arange(out_n, dtype=lucid.Float32, device=device)
                + 0.5) * (in_n / out_n) - 0.5

    indices_d = _idx(out_d, D, align_corners).clip(0, D - 1)
    indices_h = _idx(out_h, H, align_corners).clip(0, H - 1)
    indices_w = _idx(out_w, W, align_corners).clip(0, W - 1)

    d0_f = lucid.floor(indices_d)
    h0_f = lucid.floor(indices_h)
    w0_f = lucid.floor(indices_w)

    d0 = d0_f.astype(lucid.Int)
    d1 = (d0_f + 1).clip(0, D - 1).astype(lucid.Int)
    h0 = h0_f.astype(lucid.Int)
    h1 = (h0_f + 1).clip(0, H - 1).astype(lucid.Int)
    w0 = w0_f.astype(lucid.Int)
    w1 = (w0_f + 1).clip(0, W - 1).astype(lucid.Int)

    d_lerp = indices_d - d0_f
    h_lerp = indices_h - h0_f
    w_lerp = indices_w - w0_f

    c000 = input_[:, :, d0[:, None, None], h0[None, :, None], w0[None, None, :]]
    c001 = input_[:, :, d0[:, None, None], h0[None, :, None], w1[None, None, :]]
    c010 = input_[:, :, d0[:, None, None], h1[None, :, None], w0[None, None, :]]
    c011 = input_[:, :, d0[:, None, None], h1[None, :, None], w1[None, None, :]]
    c100 = input_[:, :, d1[:, None, None], h0[None, :, None], w0[None, None, :]]
    c101 = input_[:, :, d1[:, None, None], h0[None, :, None], w1[None, None, :]]
    c110 = input_[:, :, d1[:, None, None], h1[None, :, None], w0[None, None, :]]
    c111 = input_[:, :, d1[:, None, None], h1[None, :, None], w1[None, None, :]]

    c00 = c000 * (1 - w_lerp) + c001 * w_lerp
    c01 = c010 * (1 - w_lerp) + c011 * w_lerp
    c10 = c100 * (1 - w_lerp) + c101 * w_lerp
    c11 = c110 * (1 - w_lerp) + c111 * w_lerp
    c0 = c00 * (1 - h_lerp[:, None]) + c01 * h_lerp[:, None]
    c1 = c10 * (1 - h_lerp[:, None]) + c11 * h_lerp[:, None]
    return c0 * (1 - d_lerp[:, None, None]) + c1 * d_lerp[:, None, None]


def _interpolate_nearest_3d(
    input_: Tensor, size: tuple[int, int, int], align_corners: bool = False
) -> Tensor:
    _, _, D, H, W = input_.shape
    out_d, out_h, out_w = size
    if (D, H, W) == (out_d, out_h, out_w):
        return input_
    device = input_.device

    def _idx(out_n, in_n):
        return (lucid.floor(
            lucid.arange(out_n, dtype=lucid.Float32, device=device)
            * (in_n / out_n))
            .clip(0, in_n - 1).astype(lucid.Int32))

    return input_[:, :,
                  _idx(out_d, D)[:, None, None],
                  _idx(out_h, H)[None, :, None],
                  _idx(out_w, W)[None, None, :]]


def _interpolate_nearest(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    _, _, H, W = input_.shape
    out_h, out_w = size
    if (H, W) == (out_h, out_w):
        return input_
    device = input_.device
    scale_h = H / out_h
    scale_w = W / out_w
    indices_h = lucid.floor(
        lucid.arange(out_h, dtype=lucid.Float32, device=device) * scale_h
    ).clip(0, H - 1).astype(lucid.Int32)
    indices_w = lucid.floor(
        lucid.arange(out_w, dtype=lucid.Float32, device=device) * scale_w
    ).clip(0, W - 1).astype(lucid.Int32)
    return input_[:, :, indices_h[:, None], indices_w[None, :]]


def _interpolate_area(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    _, _, H, W = input_.shape
    out_h, out_w = size
    scale_h = H / out_h
    scale_w = W / out_w
    pooled = lucid.nn.functional.avg_pool2d(
        input_,
        kernel_size=(int(scale_h), int(scale_w)),
        stride=(int(scale_h), int(scale_w)),
    )
    return pooled[:, :, :out_h, :out_w]


# --------------------------------------------------------------------------- #
# Rotate
# --------------------------------------------------------------------------- #

def rotate(
    input_: Tensor, angle: float, center: tuple[_Scalar, _Scalar] | None = None
) -> Tensor:
    import math
    N, C, H, W = input_.shape
    if center is None:
        center_x = W / 2
        center_y = H / 2
    else:
        center_x, center_y = center

    angle_rad = -angle * (math.pi / 180.0)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rot_mat = lucid.Tensor([
        [cos_a, -sin_a, center_x - cos_a * center_x + sin_a * center_y],
        [sin_a, cos_a, center_y - sin_a * center_x - cos_a * center_y],
    ])

    y_coords = lucid.arange(H)
    x_coords = lucid.arange(W)
    y_grid, x_grid = lucid.meshgrid(y_coords, x_coords, indexing="ij")
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    ones = lucid.ones_like(x_flat)
    homogen = lucid.stack([x_flat, y_flat, ones])
    new_coords = rot_mat @ homogen
    new_x = new_coords[0].reshape(H, W).clip(0, W - 1).astype(lucid.Int)
    new_y = new_coords[1].reshape(H, W).clip(0, H - 1).astype(lucid.Int)

    rotated = lucid.zeros_like(input_, device=input_.device)
    for n in range(N):
        for c in range(C):
            rotated[n, c] = input_[n, c, new_y, new_x]
    return rotated


# --------------------------------------------------------------------------- #
# One-hot
# --------------------------------------------------------------------------- #

def one_hot(
    input_: Tensor, num_classes: int = -1, dtype: Numeric | bool | None = None
) -> Tensor:
    if input_.dtype.base_dtype is not int:
        raise TypeError("one_hot only supports integer input.")
    if num_classes == -1:
        num_classes = int(lucid.max(input_).item()) + 1

    flat = input_.reshape(-1)
    N = flat.shape[0]
    out_shape = (*input_.shape, num_classes)
    out = lucid.zeros(N, num_classes, device=input_.device, dtype=lucid.Int8)
    arange = lucid.arange(N, device=input_.device, dtype=lucid.Int32)
    out[arange, flat.astype(lucid.Int)] = 1
    return (out.reshape(out_shape).astype(dtype) if dtype is not None
            else out.reshape(out_shape))
