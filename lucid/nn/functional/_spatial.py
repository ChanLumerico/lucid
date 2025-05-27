from typing import Literal

import lucid
from lucid._tensor import Tensor


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


_PaddingType = Literal["zeros", "border", "reflection"]
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

    if padding_mode == "zeros":
        ...
        # TODO: Continue from here

    if align_corners:
        ix = ((grid[..., 0] + 1) * (W_in - 1) / 2).round()
        iy = ((grid[..., 1] + 1) * (H_in - 1) / 2).round()
    else:
        ix = ((grid[..., 0] + 1) * W_in / 2).round() - 0.5
        iy = ((grid[..., 1] + 1) * H_in / 2).round() - 0.5

    ix = lucid.clip(ix, 0, W_in - 1).astype(lucid.Int)
    iy = lucid.clip(iy, 0, H_in - 1).astype(lucid.Int)

    n_idx = lucid.arange(N)[:, None, None].astype(lucid.Int)
    c_idx = lucid.arange(C)[None, :, None, None].astype(lucid.Int)

    iy = iy[:, None, :, :].repeat(C, axis=1)
    ix = ix[:, None, :, :].repeat(C, axis=1)
    n_idx = n_idx[:, None, :, :].repeat(C, axis=1)

    output = input_[n_idx, c_idx, iy, ix]
    return output
