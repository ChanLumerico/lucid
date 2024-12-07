import itertools
from typing import Tuple, Optional

import lucid
from lucid._tensor import Tensor


def unfold(
    input_: Tensor,
    filter_size: Tuple[int, ...],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
) -> Tensor:
    input_shape = input_.shape
    if len(input_shape) < 2:
        raise ValueError(
            "Input tensor must have at least 2 dimensions (N and C).",
        )

    N, C, *spatial_dims = input_shape
    D = len(spatial_dims)

    if not (len(filter_size) == len(stride) == len(padding) == len(dilation) == D):
        raise ValueError(
            "filter_size, stride, padding, and dilation must have the same "
            "length as the number of spatial dimensions."
        )

    out_dims = []
    for i in range(D):
        effective_filter_size = dilation[i] * (filter_size[i] - 1) + 1
        out_dim = (
            (spatial_dims[i] + 2 * padding[i] - effective_filter_size) // stride[i]
        ) + 1
        if out_dim <= 0:
            raise ValueError(
                f"Calculated output dimension is non-positive "
                f"for spatial dimension {i}: {out_dim}."
            )
        out_dims.append(out_dim)

    pad_config = [(0, 0), (0, 0)] + [(padding[i], padding[i]) for i in range(D)]
    padded_input = lucid.pad(input_, pad_config)

    filter_offsets = list(itertools.product(*[range(fs) for fs in filter_size]))

    patches = []
    for offset in filter_offsets:
        slices = [slice(None), slice(None)]

        for d in range(D):
            start = offset[d] * dilation[d]
            end = start + stride[d] * out_dims[d]
            step = stride[d]
            slices.append(slice(start, end, step))

        patch = padded_input[tuple(slices)]
        patch = patch.unsqueeze(axis=2 + D)
        patches.append(patch)

    col = lucid.concatenate(patches, axis=2 + D)

    new_shape = [N, C] + list(filter_size) + list(out_dims)
    col = col.reshape(new_shape)

    permute_order = [0] + list(range(2 + D, 2 + D + D)) + [1] + list(range(2, 2 + D))
    col = col.transpose(permute_order)

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
    dilation: Tuple[int, ...],
    groups: int = 1,
) -> Tensor:
    N, C_in, *input_spatial = input_.shape
    C_out, C_in_div_g, *filter_size = weight.shape
    D = len(filter_size)

    if C_in % groups != 0:
        raise ValueError("Number of input channels is not divisible by groups.")
    if C_out % groups != 0:
        raise ValueError("Number of output channels is not divisible by groups.")
    if (C_in_div_g * groups) != C_in:
        raise ValueError("Weight shape is inconsistent with given number of groups.")

    out_dims = []
    for i in range(D):
        effective_filter_size = dilation[i] * (filter_size[i] - 1) + 1
        out_dim = (
            (input_spatial[i] + 2 * padding[i] - effective_filter_size) // stride[i]
        ) + 1
        if out_dim <= 0:
            raise ValueError(
                f"Calculated output dimension is non-positive "
                + f"for dimension {i}: {out_dim}."
            )
        out_dims.append(out_dim)

    col = unfold(input_, filter_size, stride, padding, dilation)

    prod_filter_size = 1
    for fs in filter_size:
        prod_filter_size *= fs

    C_in_group = C_in // groups
    C_out_group = C_out // groups

    weight_reshape = weight.reshape(groups, C_out_group, C_in_group * prod_filter_size)

    N_out_product = N
    for od in out_dims:
        N_out_product *= od

    col_reshape = col.reshape(N_out_product, groups, C_in_group * prod_filter_size)

    out_groups = []
    for g in range(groups):
        col_g = col_reshape[:, g, :]
        w_g = weight_reshape[g]

        out_g = col_g @ w_g.T
        out_groups.append(out_g)

    out_concated = lucid.concatenate(out_groups, axis=1)
    out = out_concated.reshape([N, C_out] + out_dims)

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
    dilation: int | Tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride,)
    if isinstance(padding, int):
        padding = (padding,)
    if isinstance(dilation, int):
        dilation = (dilation,)

    return _conv(input_, weight, bias, stride, padding, dilation, groups)


def conv2d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int | Tuple[int, ...] = 1,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    return _conv(input_, weight, bias, stride, padding, dilation, groups)


def conv3d(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: int | Tuple[int, ...] = 1,
    padding: int | Tuple[int, ...] = 0,
    dilation: int | Tuple[int, ...] = 1,
    groups: int = 1,
) -> Tensor:
    if isinstance(stride, int):
        stride = (stride, stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation, dilation)

    return _conv(input_, weight, bias, stride, padding, dilation, groups)
