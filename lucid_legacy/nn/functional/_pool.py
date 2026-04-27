import math
from typing import Literal

from lucid._tensor import Tensor
from lucid.nn._kernel.pool import pool_nd_kernel


def avg_pool1d(
    input_: Tensor,
    kernel_size: int | tuple[int] = 1,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
) -> Tensor:
    return pool_nd_kernel(kernel_size, stride, padding, mode="avg")(input_)


def avg_pool2d(
    input_: Tensor,
    kernel_size: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    return pool_nd_kernel(kernel_size, stride, padding, mode="avg")(input_)


def avg_pool3d(
    input_: Tensor,
    kernel_size: int | tuple[int, int, int] = 1,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
) -> Tensor:
    return pool_nd_kernel(kernel_size, stride, padding, mode="avg")(input_)


def max_pool1d(
    input_: Tensor,
    kernel_size: int | tuple[int] = 1,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
) -> Tensor:
    return pool_nd_kernel(kernel_size, stride, padding, mode="max")(input_)


def max_pool2d(
    input_: Tensor,
    kernel_size: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    return pool_nd_kernel(kernel_size, stride, padding, mode="max")(input_)


def max_pool3d(
    input_: Tensor,
    kernel_size: int | tuple[int, int, int] = 1,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
) -> Tensor:
    return pool_nd_kernel(kernel_size, stride, padding, mode="max")(input_)


def adaptive_pool1d(
    input_: Tensor, output_size: int, avg_or_max: Literal["avg", "max"]
) -> Tensor:
    L = input_.shape[2]
    kernel_size = math.ceil(L / output_size)
    stride = math.floor(L / output_size)
    pad = max((output_size - 1) * stride + kernel_size - L, 0)

    padding = (pad // 2,)
    if avg_or_max == "avg":
        return avg_pool1d(input_, kernel_size, stride, padding)
    else:
        return max_pool1d(input_, kernel_size, stride, padding)


def adaptive_pool2d(
    input_: Tensor,
    output_size: tuple[int, int] | int,
    avg_or_max: Literal["avg", "max"],
) -> Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    _, _, H, W = input_.shape
    target_H, target_W = output_size

    kernel_h = math.ceil(H / target_H)
    kernel_w = math.ceil(W / target_W)

    stride_h = math.floor(H / target_H)
    stride_w = math.floor(W / target_W)

    pad_h = max((target_H - 1) * stride_h + kernel_h - H, 0)
    pad_w = max((target_W - 1) * stride_w + kernel_w - W, 0)

    padding = (pad_h // 2, pad_w // 2)
    if avg_or_max == "avg":
        return avg_pool2d(input_, (kernel_h, kernel_w), (stride_h, stride_w), padding)
    else:
        return max_pool2d(input_, (kernel_h, kernel_w), (stride_h, stride_w), padding)


def adaptive_pool3d(
    input_: Tensor,
    output_size: tuple[int, int, int] | int,
    avg_or_max: Literal["avg", "max"],
) -> Tensor:
    if isinstance(output_size, int):
        output_size = (output_size, output_size, output_size)

    _, _, D, H, W = input_.shape
    target_D, target_H, target_W = output_size

    kernel_d = math.ceil(D / target_D)
    kernel_h = math.ceil(H / target_H)
    kernel_w = math.ceil(W / target_W)

    stride_d = math.floor(D / target_D)
    stride_h = math.floor(H / target_H)
    stride_w = math.floor(W / target_W)

    pad_d = max((target_D - 1) * stride_d + kernel_d - D, 0)
    pad_h = max((target_H - 1) * stride_h + kernel_h - H, 0)
    pad_w = max((target_W - 1) * stride_w + kernel_w - W, 0)

    padding = (pad_d // 2, pad_h // 2, pad_w // 2)
    if avg_or_max == "avg":
        return avg_pool2d(
            input_,
            (kernel_d, kernel_h, kernel_w),
            (stride_d, stride_h, stride_w),
            padding,
        )
    else:
        return max_pool2d(
            input_,
            (kernel_d, kernel_h, kernel_w),
            (stride_d, stride_h, stride_w),
            padding,
        )
