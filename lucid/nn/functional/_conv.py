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
    C_out, _, kH, kW = weight.shape

    pad_h, pad_w = padding
    H_out = H + 2 * pad_h - (kH - 1)
    W_out = W + 2 * pad_w - (kW - 1)

    x = lucid.pad(
        input_,
        [(0, 0), (0, 0), (pad_h + 2, pad_h + 2), (pad_w + 2, pad_w + 2)],
    )
    w_flat = weight.reshape((C_out * C_in, kH, kW))
    U = G_ten.matmul(w_flat).matmul(G_ten.swapaxes(1, 0))
    U = U.reshape((C_out, C_in, 4, 4))

    nH = math.ceil(H_out / 2)
    nW = math.ceil(W_out / 2)

    out = lucid.zeros((N, C_out, 2 * nH, 2 * nW), dtype=input_.dtype)
    Bt = B_ten.swapaxes(1, 0)

    for i in range(nH):
        for j in range(nW):
            y0, x0 = 2 * i, 2 * j
            patch = x[:, :, y0 : y0 + 4, x0 : x0 + 4]
            V = Bt.matmul(patch.reshape(-1, 4, 4)).matmul(B_ten)
            V = V.reshape((N, C_in, 4, 4))

            M = (V.unsqueeze(1) * U.unsqueeze(0)).sum(axis=2)
            M_flat = M.reshape(-1, 4, 4)

            y1 = A_ten.matmul(M_flat)
            Y_flat = y1.matmul(A_ten.swapaxes(1, 0))
            Y = Y_flat.reshape((N, C_out, 2, 2))

            out[:, :, y0 : y0 + 2, x0 : x0 + 2] += Y

    out = out[:, :, :H_out, :W_out]
    if bias is not None:
        out += bias.reshape((1, C_out, 1, 1))
    return out


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
        return _winograd_conv(input_, weight, bias, padding)

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
