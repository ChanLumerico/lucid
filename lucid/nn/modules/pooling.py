"""
Pooling modules.
"""

from lucid._tensor.tensor import Tensor
from lucid._types import _Size2d, _Size3d
from lucid.nn.module import Module
from lucid.nn.functional.pooling import (
    max_pool1d,
    max_pool2d,
    max_pool3d,
    avg_pool1d,
    avg_pool2d,
    avg_pool3d,
    adaptive_avg_pool1d,
    adaptive_avg_pool2d,
    adaptive_avg_pool3d,
    adaptive_max_pool2d,
    adaptive_max_pool1d,
    adaptive_max_pool3d,
)


class MaxPool1d(Module):
    """1D max pooling."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        dilation: int = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        return max_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            False,
            self.ceil_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class MaxPool2d(Module):
    """2D max pooling."""

    def __init__(
        self,
        kernel_size: _Size2d,
        stride: _Size2d | None = None,
        padding: _Size2d = 0,
        dilation: _Size2d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        return max_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            False,
            self.ceil_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class AvgPool1d(Module):
    """1D average pooling."""

    def __init__(
        self,
        kernel_size: int,
        stride: int | None = None,
        padding: int = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool1d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AvgPool2d(Module):
    """2D average pooling."""

    def __init__(
        self,
        kernel_size: _Size2d,
        stride: _Size2d | None = None,
        padding: _Size2d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool2d(
            x,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
        )

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool1d(Module):
    """Adaptive 1D average pooling."""

    def __init__(self, output_size: int | tuple[int, ...]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_pool1d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveAvgPool2d(Module):
    """Adaptive 2D average pooling."""

    def __init__(self, output_size: _Size2d) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_pool2d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool2d(Module):
    """Adaptive 2D max pooling."""

    def __init__(
        self, output_size: _Size2d, return_indices: bool = False
    ) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_max_pool2d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class MaxPool3d(Module):
    """3D max pooling."""

    def __init__(
        self,
        kernel_size: _Size3d,
        stride: _Size3d | None = None,
        padding: _Size3d = 0,
        dilation: _Size3d = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x: Tensor) -> Tensor:
        return max_pool3d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}"
        )


class AvgPool3d(Module):
    """3D average pooling."""

    def __init__(
        self,
        kernel_size: _Size3d,
        stride: _Size3d | None = None,
        padding: _Size3d = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int | None = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, x: Tensor) -> Tensor:
        return avg_pool3d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}"


class AdaptiveAvgPool3d(Module):
    """Adaptive 3D average pooling."""

    def __init__(self, output_size: _Size3d) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_avg_pool3d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool1d(Module):
    """Adaptive 1D max pooling."""

    def __init__(self, output_size: int | tuple[int, ...], return_indices: bool = False) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_max_pool1d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool3d(Module):
    """Adaptive 3D max pooling."""

    def __init__(
        self, output_size: _Size3d, return_indices: bool = False
    ) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, x: Tensor) -> Tensor:
        return adaptive_max_pool3d(x, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"
