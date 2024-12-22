import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor
from typing import Any


__all__ = [
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
]


def _single_to_tuple(value: Any, times: int) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    return (value,) * times


class _PoolNd(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...],
        padding: int | tuple[int, ...],
        *,
        D: int
    ) -> None:
        super().__init__()
        self.kernel_size = _single_to_tuple(kernel_size, D)
        self.stride = _single_to_tuple(stride, D)
        self.padding = _single_to_tuple(padding, D)


class AvgPool1d(_PoolNd):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding, D=1)

    def forward(self, input_: Tensor) -> Tensor:
        return F.avg_pool1d(input_, self.kernel_size, self.stride, self.padding)


class AvgPool2d(_PoolNd):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding, D=2)

    def forward(self, input_: Tensor) -> Tensor:
        return F.avg_pool2d(input_, self.kernel_size, self.stride, self.padding)


class AvgPool3d(_PoolNd):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding, D=3)

    def forward(self, input_: Tensor) -> Tensor:
        return F.avg_pool3d(input_, self.kernel_size, self.stride, self.padding)


class MaxPool1d(_PoolNd):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding, D=1)

    def forward(self, input_: Tensor) -> Tensor:
        return F.max_pool1d(input_, self.kernel_size, self.stride, self.padding)


class MaxPool2d(_PoolNd):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding, D=2)

    def forward(self, input_: Tensor) -> Tensor:
        return F.max_pool2d(input_, self.kernel_size, self.stride, self.padding)


class MaxPool3d(_PoolNd):
    def __init__(
        self,
        kernel_size: int | tuple[int, ...] = 1,
        stride: int | tuple[int, ...] = 1,
        padding: int | tuple[int, ...] = 0,
    ) -> None:
        super().__init__(kernel_size, stride, padding, D=3)

    def forward(self, input_: Tensor) -> Tensor:
        return F.max_pool3d(input_, self.kernel_size, self.stride, self.padding)


class AdaptiveAvgPool1d(nn.Module):
    def __init__(self, output_size: int) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input_: Tensor) -> Tensor:
        return F.adaptive_avg_pool1d(input_, self.output_size)


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self, output_size: int | tuple[int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input_: Tensor) -> Tensor:
        return F.adaptive_avg_pool2d(input_, self.output_size)


class AdaptiveAvgPool3d(nn.Module):
    def __init__(self, output_size: int | tuple[int, int, int]) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input_: Tensor) -> Tensor:
        return F.adaptive_avg_pool3d(input_, self.output_size)
