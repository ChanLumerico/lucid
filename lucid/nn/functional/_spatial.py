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


def grid_sample(
    input_: Tensor,
    grid: Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> Tensor:
    N, C, H_in, W_in = input_.shape
    N_grid, H_out, W_out = grid.shape
    assert N == N_grid, "Batch size mismatch"

    # TODO: Implement `lucid.Tensor.round()`
