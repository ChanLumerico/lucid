from typing import Literal, Callable

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

        valid: Tensor | None = None
        if padding_mode == "border":
            ix = lucid.clip(ix, 0, W_in - 1)
            iy = lucid.clip(iy, 0, H_in - 1)
        elif padding_mode == "zeros":
            valid = (
                (ix >= 0) & (ix <= W_in - 1) & (iy >= 0) & (iy <= H_in - 1)
            ).astype(input_.dtype)
            ix = lucid.clip(ix, 0, W_in - 1)
            iy = lucid.clip(iy, 0, H_in - 1)
        else:
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")

        ix_idx = ix.astype(lucid.Int)
        iy_idx = iy.astype(lucid.Int)

        n_idx = (
            lucid.arange(N, device=input_.device).reshape(N, 1, 1, 1).astype(lucid.Int)
        )
        c_idx = (
            lucid.arange(C, device=input_.device).reshape(1, C, 1, 1).astype(lucid.Int)
        )

        n_idx = n_idx.repeat(C, axis=1).repeat(H_out, axis=2).repeat(W_out, axis=3)
        c_idx = c_idx.repeat(N, axis=0).repeat(H_out, axis=2).repeat(W_out, axis=3)
        iy_idx = iy_idx[:, None, :, :].repeat(C, axis=1)
        ix_idx = ix_idx[:, None, :, :].repeat(C, axis=1)

        output = input_[n_idx, c_idx, iy_idx, ix_idx]
        if valid is not None:
            output = output * valid[:, None, :, :]
        return output

    elif mode == "bilinear":
        if padding_mode == "border":
            ix = lucid.clip(ix, 0, W_in - 1)
            iy = lucid.clip(iy, 0, H_in - 1)
        elif padding_mode != "zeros":
            raise ValueError(f"Unsupported padding_mode: {padding_mode}")

        x0 = ix.floor()
        x1 = x0 + 1
        y0 = iy.floor()
        y1 = y0 + 1

        x0_idx = lucid.clip(x0, 0, W_in - 1).astype(lucid.Int)
        x1_idx = lucid.clip(x1, 0, W_in - 1).astype(lucid.Int)
        y0_idx = lucid.clip(y0, 0, H_in - 1).astype(lucid.Int)
        y1_idx = lucid.clip(y1, 0, H_in - 1).astype(lucid.Int)

        wa = (x1 - ix) * (y1 - iy)
        wb = (x1 - ix) * (iy - y0)
        wc = (ix - x0) * (y1 - iy)
        wd = (ix - x0) * (iy - y0)

        n_idx = (
            lucid.arange(N, device=input_.device).reshape(N, 1, 1, 1).astype(lucid.Int)
        )
        c_idx = (
            lucid.arange(C, device=input_.device).reshape(1, C, 1, 1).astype(lucid.Int)
        )

        n_idx = n_idx.repeat(C, axis=1).repeat(H_out, axis=2).repeat(W_out, axis=3)
        c_idx = c_idx.repeat(N, axis=0).repeat(H_out, axis=2).repeat(W_out, axis=3)

        def _gather(y: Tensor, x: Tensor) -> Tensor:
            y = y[:, None, :, :].repeat(C, axis=1)
            x = x[:, None, :, :].repeat(C, axis=1)

            return input_[n_idx, c_idx, y, x]

        Ia = _gather(y0_idx, x0_idx)
        Ib = _gather(y1_idx, x0_idx)
        Ic = _gather(y0_idx, x1_idx)
        Id = _gather(y1_idx, x1_idx)

        if padding_mode == "zeros":
            mask_a = (
                (x0 >= 0) & (x0 <= W_in - 1) & (y0 >= 0) & (y0 <= H_in - 1)
            ).astype(input_.dtype)
            mask_b = (
                (x0 >= 0) & (x0 <= W_in - 1) & (y1 >= 0) & (y1 <= H_in - 1)
            ).astype(input_.dtype)
            mask_c = (
                (x1 >= 0) & (x1 <= W_in - 1) & (y0 >= 0) & (y0 <= H_in - 1)
            ).astype(input_.dtype)
            mask_d = (
                (x1 >= 0) & (x1 <= W_in - 1) & (y1 >= 0) & (y1 <= H_in - 1)
            ).astype(input_.dtype)

            Ia = Ia * mask_a[:, None, :, :]
            Ib = Ib * mask_b[:, None, :, :]
            Ic = Ic * mask_c[:, None, :, :]
            Id = Id * mask_d[:, None, :, :]

        wa = wa[:, None, :, :].repeat(C, axis=1)
        wb = wb[:, None, :, :].repeat(C, axis=1)
        wc = wc[:, None, :, :].repeat(C, axis=1)
        wd = wd[:, None, :, :].repeat(C, axis=1)

        output = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return output

    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
