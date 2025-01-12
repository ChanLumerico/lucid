from typing import Literal, ClassVar, Type

import lucid
import lucid.nn as nn
from lucid._tensor import Tensor


__all__ = [
    "ConvBNReLU1d",
    "ConvBNReLU2d",
    "ConvBNReLU3d",
    "DepthSeparableConv1d",
    "DepthSeparableConv2d",
    "DepthSeparableConv3d",
    "SEModule",
    "SelectiveKernel",
]


_PaddingStr = Literal["same", "valid"]

_Conv = [nn.Conv1d, nn.Conv2d, nn.Conv3d]
_BN = [nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d]


class _ConvBNReLU(nn.Module):
    D: ClassVar[int | None] = None

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
    ) -> None:
        super().__init__()
        if self.D is None:
            raise ValueError("Must specify 'D' value.")

        self.conv: nn.Module = _Conv[self.D - 1](
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            conv_bias,
        )
        self.bn: nn.Module = _BN[self.D - 1](
            out_channels, eps, momentum, bn_affine, track_running_stats
        )
        self.relu = nn.ReLU()

    def forward(self, input_: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(input_)))


class ConvBNReLU1d(_ConvBNReLU):
    D: ClassVar[int] = 1


class ConvBNReLU2d(_ConvBNReLU):
    D: ClassVar[int] = 2


class ConvBNReLU3d(_ConvBNReLU):
    D: ClassVar[int] = 3


class _DepthSeparableConv(nn.Module):
    D: ClassVar[int | None] = None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        padding: _PaddingStr | int | tuple[int, ...] = 0,
        dilation: int | tuple[int, ...] = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if self.D is None:
            raise ValueError("Must specify 'D' value.")

        self.depthwise = _Conv[self.D - 1](
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = _Conv[self.D - 1](
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,
        )
        self.reversed = reversed

    def forward(self, input_: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(input_))


class DepthSeparableConv1d(_DepthSeparableConv):
    D: ClassVar[int] = 1


class DepthSeparableConv2d(_DepthSeparableConv):
    D: ClassVar[int] = 2


class DepthSeparableConv3d(_DepthSeparableConv):
    D: ClassVar[int] = 3


class SEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Only supports 4D-tensors.")

        spatial_axes = (-1, -2)
        y = self.avgpool(x).squeeze(axis=spatial_axes)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))

        y = y.unsqueeze(axis=spatial_axes)
        out = x * y
        return out


class SelectiveKernel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        stride: int = 1,
        padding: _PaddingStr | None = None,
        groups: int = 1,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        branches = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=ks,
                stride=stride,
                padding=(ks // 2 if padding is None else padding),
                groups=groups,
                bias=False,
            )
            for ks in kernel_sizes
        ]
        self.branches = nn.ModuleList(branches)

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(
                out_channels, out_channels // reduction, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels // reduction),
            nn.ReLU(),
            nn.Conv2d(
                out_channels // reduction,
                len(kernel_sizes),
                kernel_size=1,
                bias=False,
            ),
        )

        self.softmax = nn.Softmax(axis=1)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError("Only supports 4D-tensors.")

        branch_outs = [branch(x) for branch in self.branches]
        branch_outs = lucid.stack(branch_outs, axis=1)

        att_scores = self.attention(branch_outs.sum(axis=1))
        att_weights = self.softmax(att_scores).unsqueeze(axis=2)

        out = (branch_outs * att_weights).sum(axis=1)
        return out
