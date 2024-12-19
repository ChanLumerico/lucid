import lucid

from lucid._tensor import Tensor


def _interpolate_bilinear(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    N, C, H, W = input_.shape
    out_h, out_w = size

    if H == out_h and W == out_w:
        return input_

    scale_h = H / out_h
    scale_w = W / out_w

    indices_h = (lucid.arange(out_h) * scale_h).astype(int)  # .clip(0, H - 1)

    """Complete this function."""
