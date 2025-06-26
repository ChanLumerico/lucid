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
        raise ValueError("Input tensor must have at least 2 dimensions (N and C).")
    N, C, *spatial_dims = input_shape
    D = len(spatial_dims)

    if not (len(filter_size) == len(stride) == len(padding) == len(dilation) == D):
        raise ValueError(
            "filter_size, stride, padding, and dilation must match spatial dims."
        )

    out_dims = []
    for i in range(D):
        eff_k = dilation[i] * (filter_size[i] - 1) + 1
        o = (spatial_dims[i] + 2 * padding[i] - eff_k) // stride[i] + 1
        if o <= 0:
            raise ValueError(f"Non-positive output dim for axis {i}: {o}")
        out_dims.append(o)

    pad_config = [(0, 0), (0, 0)] + [(padding[i], padding[i]) for i in range(D)]
    x = lucid.pad(input_, pad_config)

    offsets = list(itertools.product(*[range(k) for k in filter_size]))
    patches = []
    for off in offsets:
        sl = [slice(None), slice(None)]
        for d in range(D):
            start = off[d] * dilation[d]
            end = start + stride[d] * out_dims[d]
            sl.append(slice(start, end, stride[d]))

        p = x[tuple(sl)]
        p = p.unsqueeze(axis=2)
        patches.append(p)

    col = lucid.concatenate(patches, axis=2)
    new_shape = [N, C] + list(filter_size) + out_dims
    col = col.reshape(new_shape)

    perm = [0] + list(range(2 + D, 2 + 2 * D)) + [1] + list(range(2, 2 + D))
    col = col.transpose(perm)

    N_out = N
    for o in out_dims:
        N_out *= o
    C_filt = C
    for k in filter_size:
        C_filt *= k

    return col.reshape((N_out, C_filt))


def _im2col_conv(
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

    if C_in % groups != 0 or C_out % groups != 0 or C_in_div_g * groups != C_in:
        raise ValueError("Inconsistent channel/group configuration.")

    out_dims = []
    for i in range(D):
        eff = dilation[i] * (filter_size[i] - 1) + 1
        o = (input_spatial[i] + 2 * padding[i] - eff) // stride[i] + 1
        if o <= 0:
            raise ValueError(f"Non-positive output dim for axis {i}: {o}")
        out_dims.append(o)

    col = unfold(input_, filter_size, stride, padding, dilation)

    prod_filter = 1
    for k in filter_size:
        prod_filter *= k

    C_in_g = C_in // groups
    C_out_g = C_out // groups

    weight_rs = weight.reshape(groups, C_out_g, C_in_g * prod_filter)
    N_out = N
    for o in out_dims:
        N_out *= o
    col_rs = col.reshape(N_out, groups, C_in_g * prod_filter)

    outs = []
    for g in range(groups):
        c_g = col_rs[:, g, :]
        w_g = weight_rs[g]
        out = lucid.einops.einsum("nk,ok->no", c_g, w_g)
        outs.append(out)

    out_cat = lucid.concatenate(outs, axis=1)

    new_shape = [N] + out_dims + [C_out]
    out_nd = out_cat.reshape(new_shape)

    perm = [0, D + 1] + list(range(1, 1 + D))
    out_final = out_nd.transpose(perm)

    if bias is not None:
        bias_sh = [1, C_out] + [1] * D
        out_final += bias.reshape(tuple(bias_sh))

    return out_final


def conv(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
) -> Tensor:
    if len(input_.shape) < 3 or len(weight.shape) < 3:
        raise ValueError("Input and weight tensors must have at least 3 dimensions.")

    if len(stride) != len(padding) or len(stride) != len(dilation):
        raise ValueError("Stride, padding, and dilation must have the same length.")

    return _im2col_conv(input_, weight, bias, stride, padding, dilation, groups)


def _upsample_nd(input_: Tensor, stride: Tuple[int, ...]) -> Tensor:
    x = input_
    D = len(stride)
    for d in range(D):
        axis = d + 2
        s = stride[d]
        if s <= 1:
            continue

        patches = []
        L = x.shape[axis]
        for i in range(L):
            sl = [slice(None)] * x.ndim
            sl[axis] = slice(i, i + 1)
            patches.append(x[tuple(sl)])

            if i < L - 1:
                zero_shape = list(x.shape)
                zero_shape[axis] = s - 1
                patches.append(lucid.zeros(*zero_shape, device=x.device))

        x = lucid.concatenate(patches, axis=axis)

    return x


def conv_transpose(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    output_padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int = 1,
) -> Tensor:
    C_in = input_.shape[1]
    C_in_w, C_out_g, *kernel_size = weight.shape
    D = len(kernel_size)

    if len(output_padding) != D:
        raise ValueError(
            f"output_padding length must be {D}, got {len(output_padding)}"
        )
    for i, op in enumerate(output_padding):
        if op < 0 or op >= stride[i]:
            raise ValueError(
                f"output_padding[{i}] must be in range [0, {stride[i]}), got {op}"
            )
    if C_in_w != C_in:
        raise ValueError("Weight's first dimension must match input channels.")
    if C_in % groups != 0:
        raise ValueError("Input channels must be divisible by groups.")
    C_in_g = C_in // groups

    pad_ = tuple(dilation[i] * (kernel_size[i] - 1) - padding[i] for i in range(D))

    outputs = []
    for g in range(groups):
        inp_g = input_[:, g * C_in_g : (g + 1) * C_in_g]
        w_seg = weight[g * C_in_g : (g + 1) * C_in_g]

        perm = [1, 0] + list(range(2, 2 + D))
        w_t = w_seg.transpose(perm)
        flip_slices = [slice(None), slice(None)] + [
            slice(None, None, -1) for _ in range(D)
        ]
        w_t = w_t[tuple(flip_slices)]
        ups = _upsample_nd(inp_g, stride)

        if any(op > 0 for op in output_padding):
            for d, op in enumerate(output_padding):
                if op > 0:
                    axis = d + 2
                    zero_shape = list(ups.shape)
                    zero_shape[axis] = op
                    zeros = lucid.zeros(*zero_shape, dtype=ups.dtype, device=ups.device)
                    ups = lucid.concatenate([ups, zeros], axis=axis)

        out_g = _im2col_conv(
            ups,
            w_t,
            bias=None,
            stride=(1,) * D,
            padding=pad_,
            dilation=dilation,
            groups=1,
        )
        outputs.append(out_g)

    output = lucid.concatenate(outputs, axis=1)
    if bias is not None:
        b_shape = [1, C_out_g * groups] + [1] * D
        output = output + bias.reshape(tuple(b_shape))

    return output
