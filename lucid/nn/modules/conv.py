"""
Convolution and transposed convolution modules.
"""

import math
from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
# F imported lazily inside forward()
import lucid.nn.init as init


def _pair(v: int | tuple[int, int]) -> tuple[int, int]:
    return (v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


def _triple(v: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (v, v, v) if isinstance(v, int) else tuple(v)  # type: ignore[return-value]


class Conv1d(Module):
    """1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = Parameter(empty(out_channels, in_channels // groups, kernel_size, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.conv1d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class Conv2d(Module):
    """2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.weight = Parameter(empty(out_channels, in_channels // groups, kh, kw, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        self._init_weights()

    def _init_weights(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class Conv3d(Module):
    """3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        kd, kh, kw = _triple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        self.weight = Parameter(empty(out_channels, in_channels // groups, kd, kh, kw, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.conv3d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class ConvTranspose1d(Module):
    """Transposed 1D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.weight = Parameter(empty(in_channels, out_channels // groups, kernel_size, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class ConvTranspose2d(Module):
    """Transposed 2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] = 0,
        output_padding: int | tuple[int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] = 1,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        kh, kw = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.output_padding = _pair(output_padding)
        self.groups = groups
        self.dilation = _pair(dilation)
        self.weight = Parameter(empty(in_channels, out_channels // groups, kh, kw, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")


class ConvTranspose3d(Module):
    """Transposed 3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
        output_padding: int | tuple[int, int, int] = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int, int] = 1,
        device: Any = None,
        dtype: Any = None,
    ) -> None:
        super().__init__()
        kd, kh, kw = _triple(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.output_padding = _triple(output_padding)
        self.groups = groups
        self.dilation = _triple(dilation)
        self.weight = Parameter(empty(in_channels, out_channels // groups, kd, kh, kw, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Any) -> Any:
        from lucid.nn import functional as F
        return F.conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")
