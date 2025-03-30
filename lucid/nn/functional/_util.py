import lucid
import lucid.nn.functional

from lucid._tensor import Tensor
from lucid.types import _Scalar


def _interpolate_bilinear(
    input_: Tensor, size: tuple[int, int], align_corners: bool = False
) -> Tensor:
    _, _, H, W = input_.shape
    out_h, out_w = size

    scale_h = (H - 1) / (out_h - 1) if align_corners else H / out_h
    scale_w = (W - 1) / (out_w - 1) if align_corners else W / out_w

    indices_h = lucid.arange(out_h).to(input_.device) * scale_h
    indices_w = lucid.arange(out_w).to(input_.device) * scale_w

    if not align_corners:
        indices_h += 0.5 * scale_h
        indices_w += 0.5 * scale_w

    indices_h = indices_h.clip(0, H - 1)
    indices_w = indices_w.clip(0, W - 1)

    top_indices = indices_h.astype(lucid.Int)
    bot_indices = (top_indices + 1).clip(0, H - 1).astype(lucid.Int)
    left_indices = indices_w.astype(lucid.Int)
    right_indices = (left_indices + 1).clip(0, W - 1).astype(lucid.Int)

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


def rotate(
    input_: Tensor, angle: float, center: tuple[_Scalar, _Scalar] | None = None
) -> Tensor:
    N, C, H, W = input_.shape

    if center is None:
        center_x = W / 2
        center_y = H / 2
    else:
        center_x, center_y = center

    angle_rad = -angle * (lucid.pi / 180)
    cos_a = lucid.cos(angle_rad).data
    sin_a = lucid.sin(angle_rad).data

    rot_mat = [
        [cos_a, -sin_a, center_x - cos_a * center_x + sin_a * center_y],
        [sin_a, cos_a, center_y - sin_a * center_x - cos_a * center_y],
    ]
    rot_mat = lucid.to_tensor(rot_mat)

    y_coords, x_coords = lucid.arange(H), lucid.arange(W)
    y_grid, x_grid = lucid.meshgrid(y_coords, x_coords, indexing="ij")

    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()

    ones = lucid.ones_like(x_flat)
    homogen_coords = lucid.stack([x_flat, y_flat, ones])

    new_coords = (rot_mat @ homogen_coords).free()
    new_x = new_coords[0].reshape(H, W)
    new_y = new_coords[1].reshape(H, W)

    new_x = new_x.clip(0, W - 1).astype(lucid.Int)
    new_y = new_y.clip(0, H - 1).astype(lucid.Int)

    rotated_img = lucid.zeros_like(input_, device=input_.device)
    for n in range(N):
        for c in range(C):
            rotated_img[n, c] = input_[n, c, new_y, new_x]

    return rotated_img


def embedding(
    input_: Tensor,
    weight: Tensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
) -> Tensor:
    output = weight[input_.astype(lucid.Int)]
    if padding_idx is not None:
        mask = input_.data == padding_idx
        output *= 1 - mask[..., None]

    if max_norm is not None:
        norm = (output**norm_type).sum(axis=-1, keepdims=True) ** (1 / norm_type)
        scaling = max_norm / (norm + (norm == 0))
        output *= scaling

    return output
