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
_G = [[0.25, 0, 0], [-0.25, -0.25, -0.25], [-0.25, 0.25, -0.25], [0.25, 0, 0]]
_A = [[1, 1, 1, 0], [0, 1, -1, -1]]

B_ten = Tensor(_B, dtype=float)
G_ten = Tensor(_G, dtype=float)
A_ten = Tensor(_A, dtype=float)


def _winograd_conv(input_: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
    N, C_in, H, W = input_.shape
    C_out = weight.shape[0]

    extra_h, extra_w = H % 2, W % 2
    w_flat = weight.reshape((C_out * C_in, 3, 3))
    Gt = G_ten.swapaxes(1, 0)
    w_t = G_ten.matmul(w_flat).matmul(Gt)
    W_win = w_t.reshape((C_out, C_in, 4, 4))

    pad_h = (1, 1 + extra_h)
    pad_w = (1, 1 + extra_w)
    x_pad = lucid.pad(input_, [(0, 0), (0, 0), pad_h, pad_w])

    H_pad = H + 2 + extra_h
    W_pad = W + 2 + extra_w
    out_pad = lucid.zeros((N, C_out, H_pad - 2, W_pad - 2), dtype=input_.dtype)

    Bt = B_ten.swapaxes(1, 0)
    nH = math.ceil(H / 2)
    nW = math.ceil(W / 2)

    for i in range(nH):
        for j in range(nW):
            h0, w0 = i * 2, j * 2
            d = x_pad[:, :, h0 : h0 + 4, w0 : w0 + 4]
            d_flat = d.reshape((-1, 4, 4))
            d_t = Bt.matmul(d_flat).matmul(B_ten)

            D_win = d_t.reshape((N, C_in, 4, 4))
            M_sum = (D_win.unsqueeze(1) * W_win.unsqueeze(0)).sum(axis=2)
            m_flat = M_sum.reshape((-1, 4, 4))

            At = A_ten.swapaxes(1, 0)
            y_flat = At.matmul(m_flat).matmul(A_ten)

            Y = y_flat.reshape((N, C_out, 2, 2))
            out_pad[:, :, h0 : h0 + 2, w0 : w0 + 2] += Y

    if bias is not None:
        out_pad[:, :, :H, :W] += bias.reshape((1, C_out, 1, 1))

    return out_pad[:, :, :H, :W]


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
        and padding == (1, 1)
        and dilation == (1, 1)
        and groups == 1
    ):
        return _winograd_conv(input_, weight, bias)

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
