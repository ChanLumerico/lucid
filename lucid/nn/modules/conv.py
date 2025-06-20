import math
from typing import Any, Literal

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


__all__ = ["Unfold", "Conv1d", "Conv2d", "Conv3d"]


def _single_to_tuple(value: Any, times: int) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,) * times


class Unfold(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...],
        dilation: int | tuple[int, ...] = 1,
    ) -> None:
        super().__init__()
        self.kernel_size = _single_to_tuple(kernel_size, 2)
        self.stride = _single_to_tuple(stride, 2)
        self.padding = _single_to_tuple(padding, 2)
        self.dilation = _single_to_tuple(dilation, 2)

    def forward(self, input_: Tensor) -> Tensor:
        if input_.ndim != 4:
            raise ValueError(f"'nn.Unfold' only supports 4D-tensors. (i.e. images)")

        unfolded = F.unfold(
            input_, self.kernel_size, self.stride, self.padding, self.dilation
        )

        N, C, *spatial_dims = input_.shape
        out_dims = []
        for in_dim, p, d, k, s in zip(
            spatial_dims, self.padding, self.dilation, self.kernel_size, self.stride
        ):
            eff_k = d * (k - 1) + 1
            out_dim = math.floor((in_dim + 2 * p - eff_k) / s + 1)
            out_dims.append(out_dim)

        L = math.prod(out_dims)
        Ck = C * math.prod(self.kernel_size)
        return unfolded.reshape(N, Ck, L)


_PaddingStr = Literal["same", "valid"]


class _ConvNd(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: _PaddingStr | int | tuple[int, ...],
        dilation: int | tuple[int, ...],
        groups: int,
        bias: bool,
        *,
        D: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.groups = groups

        self.kernel_size = _single_to_tuple(kernel_size, D)
        self.stride = _single_to_tuple(stride, D)
        self.dilation = _single_to_tuple(dilation, D)

        if isinstance(padding, str):
            if padding == "same":
                self.padding = tuple(
                    (self.dilation[i] * (self.kernel_size[i] - 1)) // 2
                    for i in range(D)
                )
            elif padding == "valid":
                self.padding = (0,) * D
            else:
                raise ValueError(f"Unknown padding string: {padding}")
        else:
            self.padding = _single_to_tuple(padding, D)

        if groups <= 0:
            raise ValueError("groups must be a positive integer.")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        if out_channels % groups != 0:
            raise ValueError("out_channels mube be divisible by groups.")

        weight_ = lucid.empty(out_channels, in_channels // groups, *self.kernel_size)
        self.weight = nn.Parameter(weight_)

        if bias:
            bias_ = lucid.empty((out_channels,))
            self.bias = nn.Parameter(bias_)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform(self.weight)

        if self.bias is not None:
            fan_in, _ = nn.init._dist._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += f", padding={self.padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"

        return s


class Conv1d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            D=1,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        lucid._check_input_dim(input_, dim=3)
        return F.conv1d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self.weight, self.bias)


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            D=2,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        lucid._check_input_dim(input_, dim=4)
        return F.conv2d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self.weight, self.bias)


class Conv3d(_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            D=3,
        )

    def _conv_forward(
        self, input_: Tensor, weight: Tensor, bias: Tensor | None
    ) -> Tensor:
        lucid._check_input_dim(input_, dim=5)
        return F.conv3d(
            input_, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input_: Tensor) -> Tensor:
        return self._conv_forward(input_, self.weight, self.bias)
