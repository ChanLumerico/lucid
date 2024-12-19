import lucid

from lucid._tensor import Tensor
import lucid.nn.functional


def _interpolate_bilinear(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    _, _, H, W = input_.shape
    out_h, out_w = size

    scale_h = (H - 1) / (out_h - 1) if align_corners else H / out_h
    scale_w = (W - 1) / (out_w - 1) if align_corners else W / out_w

    indices_h = lucid.arange(out_h) * scale_h
    indices_w = lucid.arange(out_w) * scale_w

    if not align_corners:
        indices_h += 0.5 * scale_h
        indices_w += 0.5 * scale_w

    indices_h = indices_h.clip(0, H - 1)
    indices_w = indices_w.clip(0, W - 1)

    top_indices = indices_h.astype(int)
    bot_indices = (top_indices + 1).clip(0, H - 1).astype(int)
    left_indices = indices_w.astype(int)
    right_indices = (left_indices + 1).clip(0, W - 1).astype(int)

    h_lerp = indices_h - top_indices
    w_lerp = indices_w - left_indices

    top_left = input_[:, :, top_indices[:, None], left_indices]
    top_right = input_[:, :, top_indices[:, None], right_indices]
    bot_left = input_[:, :, bot_indices[:, None], left_indices]
    bot_right = input_[:, :, bot_indices[:, None], right_indices]

    top = top_left * (1 - w_lerp) + top_right * w_lerp
    bot = bot_left * (1 - w_lerp) + bot_right * w_lerp

    interpolated = top * (1 - h_lerp[:, None]) + bot * h_lerp[:, None]
    return interpolated


def _interpolate_nearest(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> None:
    _, _, H, W = input_.shape
    out_h, out_w = size

    scale_h = H / out_h
    scale_w = W / out_w

    indices_h = (lucid.arange(out_h) * scale_h).clip(0, H - 1).astype(int)
    indices_w = (lucid.arange(out_w) * scale_w).clip(0, W - 1).astype(int)

    return input_[:, :, indices_h[:, None], indices_w]


def _interpolate_area(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> None:
    _, _, H, W = input_.shape
    out_h, out_w = size

    scale_h = H / out_h
    scale_w = W / out_w

    pooled = lucid.nn.functional.avg_pool2d(
        input_,
        kernel_size=(int(scale_h), int(scale_w)),
        stride=(int(scale_h), int(scale_w)),
    )
    return pooled[:, :, out_h, out_w]
