import itertools
import math
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
        outs.append(c_g @ w_g.T)
    out_cat = lucid.concatenate(outs, axis=1)

    new_shape = [N] + out_dims + [C_out]
    out_nd = out_cat.reshape(new_shape)

    perm = [0, D + 1] + list(range(1, 1 + D))
    out_final = out_nd.transpose(perm)

    if bias is not None:
        bias_sh = [1, C_out] + [1] * D
        out_final += bias.reshape(tuple(bias_sh))

    return out_final


_B = [[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]
_G = [[1, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]
_A = [[1, 1, 1, 0], [0, 1, -1, -1]]

B_ten = Tensor(_B, dtype=float)
G_ten = Tensor(_G, dtype=float)
A_ten = Tensor(_A, dtype=float)


def _winograd_conv(
    input_: Tensor, weight: Tensor, bias: Optional[Tensor], padding: Tuple[int, int]
) -> Tensor:
    N, C_in, H, W = input_.shape
    C_out, _, kh, kw = weight.shape

    pad_h, pad_w = padding
    assert kh == 3 and kw == 3, "Kernel size must be 3x3 for Winograd Convolution."

    x_pad = lucid.pad(input_, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    H_out = H + 2 * pad_h - kh + 1
    W_out = W + 2 * pad_w - kw + 1

    m, r = 2, 3
    alpha = m + r - 1
    nH = int(math.ceil(H_out / m))
    nW = int(math.ceil(W_out / m))

    H_pad = nH * m + r - 1
    W_pad = nW * m + r - 1

    extra_h = H_pad - (H + 2 * pad_h)
    extra_w = W_pad - (W + 2 * pad_w)
    if extra_h > 0 or extra_w > 0:
        x_pad = lucid.pad(x_pad, ((0, 0), (0, 0), (0, extra_h), (0, extra_w)))

    U = lucid.einops.einsum("ik, ockl, jl -> ocij", G_ten, weight, G_ten)
    Y = lucid.zeros((N, C_out, nH * m, nW * m), dtype=input_.dtype)

    for i in range(nH):
        for j in range(nW):
            d = x_pad[:, :, i * m : i * m + alpha, j * m : j * m + alpha]
            d_flat = d.reshape(-1, alpha, alpha)

            V_flat = B_ten @ d_flat @ B_ten.T
            V = V_flat.reshape(N, C_in, alpha, alpha)

            M = lucid.einops.einsum("ocij, ncij -> noij", U, V)
            M_flat = M.reshape(-1, alpha, alpha)

            # TODO: implement `lucid.tensordot`


def _conv(
    input_: Tensor,
    weight: Tensor,
    bias: Optional[Tensor],
    stride: Tuple[int, ...],
    padding: Tuple[int, ...],
    dilation: Tuple[int, ...],
    groups: int,
) -> Tensor:
    if (
        input_.ndim == 4
        and weight.shape[2:] == (3, 3)
        and stride == (1, 1)
        and dilation == (1, 1)
        and groups == 1
    ):
        # return _winograd_conv(input_, weight, bias, padding)
        # NOTE: Implement Winograd convolution first before enabling this.
        pass

    if len(input_.shape) < 3 or len(weight.shape) < 3:
        raise ValueError("Input and weight tensors must have at least 3 dimensions.")

    if len(stride) != len(padding) or len(stride) != len(dilation):
        raise ValueError("Stride, padding, and dilation must have the same length.")

    return _im2col_conv(input_, weight, bias, stride, padding, dilation, groups)


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
