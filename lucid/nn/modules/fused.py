from typing import Literal

import lucid.nn as nn
from lucid._tensor import Tensor

from lucid.nn.modules.conv import Conv1d, Conv2d, Conv3d
from lucid.nn.modules.norm import BatchNorm1d, BatchNorm2d, BatchNorm3d


__all__ = ["ConvBNReLU1d", "ConvBNReLU2d", "ConvBNReLU3d"]


_PaddingStr = Literal["same", "valid"]

_Conv = [Conv1d, Conv2d, Conv3d]
_BN = [BatchNorm1d, BatchNorm2d, BatchNorm3d]


class _ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        groups: int = 1,
        conv_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        bn_affine: bool = True,
        track_running_stats: bool = True,
        /,
        D: int | None = None,
    ) -> None:
        super().__init__()
        if D is None:
            raise ValueError("Must specify 'D' value.")

        self.conv: nn.Module = _Conv[D - 1](
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_bias,
        )
        self.bn: nn.Module = _BN[D - 1](
            out_channels, eps, momentum, bn_affine, track_running_stats
        )
        self.relu = nn.ReLU()

    def forward(self, input_: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(input_)))


class ConvBNReLU1d(_ConvBNReLU):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int],
        stride: int | tuple[int] = 1,
        padding: _PaddingStr | int | tuple[int] = 0,
        dilation: int | tuple[int] = 1,
        groups: int = 1,
        conv_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        bn_affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_bias,
            eps,
            momentum,
            bn_affine,
            track_running_stats,
            D=1,
        )


class ConvBNReLU2d(_ConvBNReLU):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        padding: _PaddingStr | int | tuple[int, int] = 0,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        conv_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        bn_affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_bias,
            eps,
            momentum,
            bn_affine,
            track_running_stats,
            D=2,
        )


class ConvBNReLU3d(_ConvBNReLU):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: _PaddingStr | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        groups: int = 1,
        conv_bias: bool = True,
        eps: float = 1e-5,
        momentum: float | None = 0.1,
        bn_affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_bias,
            eps,
            momentum,
            bn_affine,
            track_running_stats,
            D=3,
        )
