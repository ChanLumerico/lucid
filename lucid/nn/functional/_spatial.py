from typing import Literal, Callable

import lucid
from lucid._tensor import Tensor

from ._util import _interpolate_nearest


def affine_grid(
    theta: Tensor, size: tuple[int, ...], align_corners: bool = True
) -> Tensor:
    N, _, H, W = size
    device = theta.device

    if align_corners:
        xs = lucid.linspace(-1, 1, W)
        ys = lucid.linspace(-1, 1, H)
    else:
        xs = lucid.linspace(-1 + 1 / W, 1 - 1 / W, W)
        ys = lucid.linspace(-1 + 1 / H, 1 - 1 / H, H)

    x, y = lucid.meshgrid(xs, ys)
    ones = lucid.ones_like(x)

    grid = lucid.stack([x, y, ones], axis=-1)
    grid = grid.reshape(1, H * W, 3).repeat(N, axis=0)
    grid = grid.astype(lucid.Float).to(device).free()

    theta = theta.reshape(N, 2, 3)
    out = grid @ theta.transpose((0, 2, 1))
    out = out.reshape(N, H, W, 2)

    return out


_PaddingType = Literal["zeros", "border"]
_InterpolateType = Literal["bilinear", "nearest"]


def grid_sample(
    input_: Tensor,
    grid: Tensor,
    mode: _InterpolateType = "bilinear",
    padding_mode: _PaddingType = "zeros",
    align_corners: bool = True,
) -> Tensor:
    N, C, H_in, W_in = input_.shape
    N_grid, H_out, W_out, _ = grid.shape
    assert N == N_grid, "Batch size mismatch"

    if align_corners:
        ix = (grid[..., 0] + 1) * (W_in - 1) / 2
        iy = (grid[..., 1] + 1) * (H_in - 1) / 2
    else:
        ix = (grid[..., 0] + 1) * W_in / 2 - 0.5
        iy = (grid[..., 1] + 1) * H_in / 2 - 0.5

    if mode == "nearest":
        ix = ix.round()
        iy = iy.round()

    if padding_mode == "zeros":
        input_ = input_.pad(((0, 0), (0, 0), (1, 1), (1, 1)))
        ix = ix + 1
        iy = iy + 1
    elif padding_mode == "border":
        input_ = _interpolate_nearest(input_, size=(H_in + 2, W_in + 2))
        ix = ix + 1
        iy = iy + 1
    else:
        raise ValueError(f"Unsupported padding_mode: {padding_mode}")

    if mode == "nearest":
        ix = ix.round()
        iy = iy.round()

        ix = lucid.clip(ix, 0, input_.shape[3] - 1).astype(lucid.Int)
        iy = lucid.clip(iy, 0, input_.shape[2] - 1).astype(lucid.Int)

        n_idx = lucid.arange(N)[:, None, None].astype(lucid.Int)
        c_idx = lucid.arange(C)[None, :, None, None].astype(lucid.Int)

        iy = iy[:, None, :, :].repeat(C, axis=1)
        ix = ix[:, None, :, :].repeat(C, axis=1)
        n_idx = n_idx[:, None, :, :].repeat(C, axis=1)

        output = input_[n_idx, c_idx, iy, ix]
        return output

    elif mode == "bilinear":
        x0 = lucid.clip(ix.floor(), 0, input_.shape[3] - 1).astype(lucid.Int)
        x1 = lucid.clip(x0 + 1, 0, input_.shape[3] - 1)

        y0 = lucid.clip(iy.floor(), 0, input_.shape[2] - 1).astype(lucid.Int)
        y1 = lucid.clip(y0 + 1, 0, input_.shape[2] - 1)

        wa = (x1 - ix) * (y1 - iy)
        wb = (x1 - ix) * (iy - y0)
        wc = (ix - x0) * (y1 - iy)
        wd = (ix - x0) * (iy - y0)

        n_idx = lucid.arange(N).reshape(N, 1, 1, 1).astype(lucid.Int)
        c_idx = lucid.arange(C).reshape(1, C, 1, 1).astype(lucid.Int)

        n_idx = n_idx.repeat(C, axis=1).repeat(H_out, axis=2).repeat(W_out, axis=3)
        c_idx = c_idx.repeat(N, axis=0).repeat(H_out, axis=2).repeat(W_out, axis=3)

        def _gather(y: Tensor, x: Tensor) -> Tensor:
            y = y[:, None, :, :].repeat(C, axis=1)
            x = x[:, None, :, :].repeat(C, axis=1)

            return input_[n_idx, c_idx, y, x]

        Ia = _gather(y0, x0)
        Ib = _gather(y1, x0)
        Ic = _gather(y0, x1)
        Id = _gather(y1, x1)

        wa = wa[:, None, :, :].repeat(C, axis=1)
        wb = wb[:, None, :, :].repeat(C, axis=1)
        wc = wc[:, None, :, :].repeat(C, axis=1)
        wd = wd[:, None, :, :].repeat(C, axis=1)

        output = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return output

    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
