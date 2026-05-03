"""
Convolution and transposed convolution modules.
"""

import math
from typing import Any
from lucid.nn.module import Module
from lucid.nn.parameter import Parameter
from lucid._factories.creation import empty
import lucid.nn.init as init
from lucid.nn.functional.conv import (
    conv1d, conv2d, conv3d,
    conv_transpose1d, conv_transpose2d, conv_transpose3d,
)


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
        padding: int | str = 0,
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
        self.dilation = dilation
        self.groups = groups
        if isinstance(padding, str):
            self._padding_str: str | None = padding.lower()
            self.padding: int = 0
        else:
            self._padding_str = None
            self.padding = padding
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

    def _compute_padding(self, x: Any) -> int:
        if self._padding_str == 'valid':
            return 0
        in_len = x.shape[2]
        stride = self.stride
        dilation = self.dilation
        ks = self.kernel_size
        out_len = (in_len + stride - 1) // stride
        pad_total = max(0, (out_len - 1) * stride + (ks - 1) * dilation + 1 - in_len)
        return (pad_total + 1) // 2

    def forward(self, x: Any) -> Any:
        padding = self._compute_padding(x) if self._padding_str else self.padding
        return conv1d(x, self.weight, self.bias, self.stride, padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}")


class Conv2d(Module):
    """2D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | str = 0,
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
        self.dilation = _pair(dilation)
        self.groups = groups
        if isinstance(padding, str):
            self._padding_str: str | None = padding.lower()
            self.padding: tuple[int, int] = (0, 0)
        else:
            self._padding_str = None
            self.padding = _pair(padding)
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

    def _compute_padding(self, x: Any) -> tuple[int, int]:
        if self._padding_str == 'valid':
            return (0, 0)
        in_h, in_w = x.shape[2], x.shape[3]
        sh, sw = self.stride
        dh, dw = self.dilation
        kh, kw = self.kernel_size
        out_h = (in_h + sh - 1) // sh
        out_w = (in_w + sw - 1) // sw
        pad_h = max(0, (out_h - 1) * sh + (kh - 1) * dh + 1 - in_h)
        pad_w = max(0, (out_w - 1) * sw + (kw - 1) * dw + 1 - in_w)
        return ((pad_h + 1) // 2, (pad_w + 1) // 2)

    def forward(self, x: Any) -> Any:
        padding = self._compute_padding(x) if self._padding_str else self.padding
        return conv2d(x, self.weight, self.bias, self.stride, padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}")


class Conv3d(Module):
    """3D convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] | str = 0,
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
        self.dilation = _triple(dilation)
        self.groups = groups
        if isinstance(padding, str):
            self._padding_str: str | None = padding.lower()
            self.padding: tuple[int, int, int] = (0, 0, 0)
        else:
            self._padding_str = None
            self.padding = _triple(padding)
        self.weight = Parameter(empty(out_channels, in_channels // groups, kd, kh, kw, dtype=dtype, device=device))
        self.bias: Parameter | None = (
            Parameter(empty(out_channels, dtype=dtype, device=device)) if bias else None
        )
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def _compute_padding(self, x: Any) -> tuple[int, int, int]:
        if self._padding_str == 'valid':
            return (0, 0, 0)
        in_d, in_h, in_w = x.shape[2], x.shape[3], x.shape[4]
        sd, sh, sw = self.stride
        dd, dh, dw = self.dilation
        kd, kh, kw = self.kernel_size
        out_d = (in_d + sd - 1) // sd
        out_h = (in_h + sh - 1) // sh
        out_w = (in_w + sw - 1) // sw
        pad_d = max(0, (out_d - 1) * sd + (kd - 1) * dd + 1 - in_d)
        pad_h = max(0, (out_h - 1) * sh + (kh - 1) * dh + 1 - in_h)
        pad_w = max(0, (out_w - 1) * sw + (kw - 1) * dw + 1 - in_w)
        return ((pad_d + 1) // 2, (pad_h + 1) // 2, (pad_w + 1) // 2)

    def forward(self, x: Any) -> Any:
        padding = self._compute_padding(x) if self._padding_str else self.padding
        return conv3d(x, self.weight, self.bias, self.stride, padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self._padding_str if self._padding_str else self.padding}")


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
        return conv_transpose1d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

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
        return conv_transpose2d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

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
        return conv_transpose3d(x, self.weight, self.bias, self.stride, self.padding, self.output_padding, self.groups, self.dilation)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}")
