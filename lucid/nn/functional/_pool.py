import math
import lucid
from lucid._tensor import Tensor


def _pad_input(input_: Tensor, padding: tuple[int, ...]) -> Tensor:
    pad_config = [(0, 0), (0, 0)] + [(pad, pad) for pad in padding]
    return lucid.pad(input_, pad_config)


def _pool1d(
    input_: Tensor,
    kernel_size: int | tuple[int],
    stride: int | tuple[int],
    padding: int | tuple[int],
) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,)
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)

    _, _, L = input_.shape
    (kernel,) = kernel_size
    (stride_,) = stride
    (pad_,) = padding

    out_L = math.floor((L + 2 * pad_ - kernel) / stride_ + 1)
    padded_input = _pad_input(input_, padding)

    patches = []
    for i in range(kernel):
        start = i
        end = start + stride_ * out_L

        patch = padded_input[..., start:end:stride_]
        patches.append(patch)

    return lucid.stack(patches, axis=-1)


def _pool2d(
    input_: Tensor,
    kernel_size: int | tuple[int, int],
    stride: int | tuple[int, int],
    padding: int | tuple[int, int],
) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    _, _, H, W = input_.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    out_H = math.floor((H + 2 * pad_h - kernel_h) / stride_h + 1)
    out_W = math.floor((W + 2 * pad_w - kernel_w) / stride_w + 1)
    padded_input = _pad_input(input_, padding)

    patches = []
    for i in range(kernel_h):
        for j in range(kernel_w):
            start_h = i
            end_h = start_h + stride_h * out_H
            start_w = j
            end_w = start_w + stride_w * out_W

            patch = padded_input[
                ...,
                start_h:end_h:stride_h,
                start_w:end_w:stride_w,
            ]
            patches.append(patch)

    return lucid.stack(patches, axis=-1)


def _pool3d(
    input_: Tensor,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int],
    padding: int | tuple[int, int, int],
) -> Tensor:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)

    _, _, D, H, W = input_.shape
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    pad_d, pad_h, pad_w = padding

    out_D = math.floor((D + 2 * pad_d - kernel_d) / stride_d + 1)
    out_H = math.floor((H + 2 * pad_h - kernel_h) / stride_h + 1)
    out_W = math.floor((W + 2 * pad_w - kernel_w) / stride_w + 1)
    padded_input = _pad_input(input_, padding)

    patches = []
    for i in range(kernel_d):
        for j in range(kernel_h):
            for k in range(kernel_w):
                start_d = i
                end_d = start_d + stride_d * out_D
                start_h = j
                end_h = start_h + stride_h * out_H
                start_w = k
                end_w = start_w + stride_w * out_W

                patch = padded_input[
                    ...,
                    start_d:end_d:stride_d,
                    start_h:end_h:stride_h,
                    start_w:end_w:stride_w,
                ]
                patches.append(patch)

    return lucid.stack(patches, axis=-1)


def avg_pool1d(
    input_: Tensor,
    kernel_size: int | tuple[int] = 1,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
) -> Tensor:
    ret = _pool1d(input_, kernel_size, stride, padding).mean(axis=-1, keepdims=True)
    return ret.reshape(*ret.shape[:-1])


def avg_pool2d(
    input_: Tensor,
    kernel_size: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    ret = _pool2d(input_, kernel_size, stride, padding).mean(axis=-1, keepdims=True)
    return ret.reshape(*ret.shape[:-1])


def avg_pool3d(
    input_: Tensor,
    kernel_size: int | tuple[int, int, int] = 1,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
) -> Tensor:
    ret = _pool3d(input_, kernel_size, stride, padding).mean(axis=-1, keepdims=True)
    return ret.reshape(*ret.shape[:-1])


def max_pool1d(
    input_: Tensor,
    kernel_size: int | tuple[int] = 1,
    stride: int | tuple[int] = 1,
    padding: int | tuple[int] = 0,
) -> Tensor:
    return lucid.max(_pool1d(input_, kernel_size, stride, padding), axis=-1)


def max_pool2d(
    input_: Tensor,
    kernel_size: int | tuple[int, int] = 1,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 0,
) -> Tensor:
    return lucid.max(_pool2d(input_, kernel_size, stride, padding), axis=-1)


def max_pool3d(
    input_: Tensor,
    kernel_size: int | tuple[int, int, int] = 1,
    stride: int | tuple[int, int, int] = 1,
    padding: int | tuple[int, int, int] = 0,
) -> Tensor:
    return lucid.max(_pool3d(input_, kernel_size, stride, padding), axis=-1)


def adaptive_avg_pool1d(input_: Tensor, output_size: int) -> Tensor:
    _, _, L = input_.shape

    kernel_size = math.ceil(L / output_size)
    stride = math.floor(L / output_size)
    pad = max((output_size - 1) * stride + kernel_size - L, 0)

    padding = (pad // 2,)
    return avg_pool1d(input_, kernel_size, stride, padding)


def adaptive_avg_pool2d(
    input_: Tensor,
    output_size: tuple[int, int] | int,
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
    return avg_pool2d(
        input_,
        (kernel_h, kernel_w),
        (stride_h, stride_w),
        padding,
    )


def adaptive_avg_pool3d(
    input_: Tensor, output_size: tuple[int, int, int] | int
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
    return avg_pool2d(
        input_,
        (kernel_d, kernel_h, kernel_w),
        (stride_d, stride_h, stride_w),
        padding,
    )
