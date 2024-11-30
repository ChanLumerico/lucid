import itertools
from typing import Tuple, Optional

import lucid
from lucid._tensor import Tensor


def im2col(
    input_: Tensor,
    filter_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
) -> Tensor:
    N, C, *spatial_dims = input_.shape
    D = len(spatial_dims)

    if not (len(filter_size) == len(stride) == len(padding) == D):
        raise ValueError(
            "filter_size, stride, and padding must have "
            + "the same length as the number of spatial dimensions."
        )

    out_dims = []
    for i in range(D):
        out_dim = (spatial_dims[i] + 2 * padding[i] - filter_size[i]) // stride[i] + 1
        if out_dim <= 0:
            raise ValueError(
                f"Calculated output dimension is non-positive for dimension {i}."
            )
        out_dims.append(out_dim)

    pad_config = [(0, 0), (0, 0)] + [(padding[i], padding[i]) for i in range(D)]
    padded_input = lucid.pad(input_, pad_config)

    col_shape = [N, C] + list(filter_size) + list(out_dims)
    col = lucid.zeros(
        col_shape,
        requires_grad=input_.requires_grad,
        dtype=input_.dtype,
    )

    ranges = [range(fs) for fs in filter_size]
    for offset in itertools.product(*ranges):

        slices = [slice(None), slice(None)]
        for d in range(D):
            start = offset[d]
            end = start + stride[d] * out_dims[d]
            step = stride[d]
            slices.append(slice(start, end, step))

        patch = padded_input[tuple(slices)]

        col_indices = [slice(None), slice(None)] + list(offset) + [slice(None)] * D
        col[tuple(col_indices)] = patch

    transpose_order = [0]
    transpose_order += list(range(2 + D, 2 + D + D))
    transpose_order += [1]
    transpose_order += list(range(2, 2 + D))

    col = col.transpose(tuple(transpose_order))

    N_times_out = N
    for od in out_dims:
        N_times_out *= od

    C_times_filter = C
    for fs in filter_size:
        C_times_filter *= fs

    col = col.reshape((N_times_out, C_times_filter))
    return col


def _conv(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
) -> Tensor:
    N, _, *input_spatial = input_.shape
    C_out, _, *filter_size = weight.shape
    D = len(filter_size)

    out_dims = []
    for i in range(D):
        out_dim = (input_spatial[i] + 2 * padding[i] - filter_size[i]) // stride[i] + 1
        out_dims.append(out_dim)

    col = im2col(input_, filter_size, stride, padding)
    weight_reshape = weight.reshape((C_out, -1))

    out = col @ weight_reshape.T
    out = out.reshape([N, C_out] + out_dims)

    if bias is not None:
        bias_shape = [1, C_out] + [1] * D
        out += bias.reshape(tuple(bias_shape))

    return out


def conv1d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int | Tuple[int, ...] = 1,
    padding: int | Tuple[int, ...] = 0,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)

    return _conv(input_, weight, bias, stride, padding)


def conv2d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int | Tuple[int, ...] = 1,
    padding: int | Tuple[int, ...] = 0,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    return _conv(input_, weight, bias, stride, padding)


def conv3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int | Tuple[int, ...] = 1,
    padding: int | Tuple[int, ...] = 0,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)

    return _conv(input_, weight, bias, stride, padding)
