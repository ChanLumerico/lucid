"""
lucid.nn.functional._conv — convolution and transposed convolution.

Routes 1:1 to engine ops. Engine `conv1d/2d/3d` natively support
stride / padding / dilation / groups; backward returns (dx, dW, db).
`unfold` is a standalone im2col op exposed by the engine.
"""

from __future__ import annotations

import lucid

from lucid._C.engine import nn as _C_nn
from lucid._tensor import Tensor
from lucid._bridge import impl_of


def unfold(
    input_: Tensor,
    filter_size: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
) -> Tensor:
    return Tensor._wrap(_C_nn.unfold(
        impl_of(input_),
        [int(k) for k in filter_size],
        [int(s) for s in stride],
        [int(p) for p in padding],
        [int(d) for d in dilation]))


def _bias_or_zeros(weight: Tensor, bias: Tensor | None) -> Tensor:
    if bias is not None:
        return bias
    return lucid.zeros((weight.shape[0],), dtype=weight.dtype, device=weight.device)


def conv(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
) -> Tensor:
    if len(input_.shape) < 3 or len(weight.shape) < 3:
        raise ValueError("Input and weight tensors must have at least 3 dimensions.")
    if len(stride) != len(padding) or len(stride) != len(dilation):
        raise ValueError("Stride, padding, and dilation must have the same length.")

    b = _bias_or_zeros(weight, bias)
    n = len(stride)
    if n == 1:
        return Tensor._wrap(_C_nn.conv1d(
            impl_of(input_), impl_of(weight), impl_of(b),
            int(stride[0]), int(padding[0]), int(dilation[0]), int(groups)))
    if n == 2:
        return Tensor._wrap(_C_nn.conv2d(
            impl_of(input_), impl_of(weight), impl_of(b),
            int(stride[0]), int(stride[1]),
            int(padding[0]), int(padding[1]),
            int(dilation[0]), int(dilation[1]), int(groups)))
    if n == 3:
        return Tensor._wrap(_C_nn.conv3d(
            impl_of(input_), impl_of(weight), impl_of(b),
            int(stride[0]), int(stride[1]), int(stride[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(dilation[0]), int(dilation[1]), int(dilation[2]),
            int(groups)))
    raise ValueError(f"Unsupported conv spatial dim: {n}")


def conv_transpose(
    input_: Tensor,
    weight: Tensor,
    bias: Tensor | None,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    output_padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int = 1,
) -> Tensor:
    # Engine conv_transpose currently does not support dilation>1 or groups>1.
    # We pass the args through; CPU/GPU paths will validate.
    if any(int(d) != 1 for d in dilation):
        raise NotImplementedError(
            "conv_transpose: dilation>1 not yet supported in C++ engine")
    if int(groups) != 1:
        raise NotImplementedError(
            "conv_transpose: groups>1 not yet supported in C++ engine")

    C_out_g = weight.shape[1]
    b = bias if bias is not None else lucid.zeros(
        (C_out_g,), dtype=weight.dtype, device=weight.device)
    n = len(stride)
    if n == 1:
        return Tensor._wrap(_C_nn.conv_transpose1d(
            impl_of(input_), impl_of(weight), impl_of(b),
            int(stride[0]), int(padding[0]), int(output_padding[0])))
    if n == 2:
        return Tensor._wrap(_C_nn.conv_transpose2d(
            impl_of(input_), impl_of(weight), impl_of(b),
            int(stride[0]), int(stride[1]),
            int(padding[0]), int(padding[1]),
            int(output_padding[0]), int(output_padding[1])))
    if n == 3:
        return Tensor._wrap(_C_nn.conv_transpose3d(
            impl_of(input_), impl_of(weight), impl_of(b),
            int(stride[0]), int(stride[1]), int(stride[2]),
            int(padding[0]), int(padding[1]), int(padding[2]),
            int(output_padding[0]), int(output_padding[1]), int(output_padding[2])))
    raise ValueError(f"Unsupported conv_transpose spatial dim: {n}")
