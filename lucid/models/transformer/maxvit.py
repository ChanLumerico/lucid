from typing import Any, override
from functools import partial

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


class _SqueezeExcite(nn.Module):
    def __init__(
        self,
        in_channels: int,
        rd_ratio: float = 0.25,
        rd_channels: int | None = None,
        act_layer: type[nn.Module] = nn.ReLU,
        gate_layer: type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()
        if rd_channels is None:
            rd_channels = int(in_channels * rd_ratio)

        self.conv_reduce = nn.Conv2d(in_channels, rd_channels, kernel_size=1)
        self.act = act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, in_channels, kernel_size=1)
        self.gate = gate_layer()

    def forward(self, x: Tensor) -> Tensor:
        x_se = x.mean(axis=(2, 3), keepdims=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)

        return x * self.gate(x_se)


class _DepthwiseSeparableConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        act_layer: type[nn.Module] = nn.ReLU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.act = act_layer()
        self.bn = norm_layer(out_channels)
        self.drop_path = (
            nn.DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.drop_path(self.act(self.bn(x)))

        return x


class _MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downscale: bool = False,
        act_layer: type[nn.module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop_path_rate = drop_path
        if not downscale:
            assert (
                in_channels == out_channels
            ), "in/out channels must be equal when downscale=True."

        self.main_path = nn.Sequential(
            norm_layer(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            _DepthwiseSeparableConv(
                in_channels,
                out_channels,
                stride=2 if downscale else 1,
                act_layer=act_layer,
                drop_path_rate=drop_path,
            ),
            _SqueezeExcite(out_channels, rd_ratio=0.25),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
        )
        self.skip_path = (
            nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
            if downscale
            else nn.Identity()
        )
        self.drop = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.main_path(x)
        out = self.drop(out)
        out += self.skip_path(x)

        return out


def _window_partition(x: Tensor, window_size: tuple[int, int] = (7, 7)) -> Tensor:
    B, C, H, W = x.shape
    windows = x.reshape(
        B, C, H // window_size[0], window_size[0], W // window_size[1], window_size[1]
    )
    windows = windows.transpose((0, 2, 4, 3, 5, 1)).reshape(-1, *window_size, C)
    return windows


def _window_reverse(
    windows: Tensor,
    original_size: tuple[int, int],
    window_size: tuple[int, int] = (7, 7),
) -> Tensor:
    H, W = original_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))

    out = windows.reshape(B, H // window_size[0], W // window_size[1], *window_size, -1)
    out = out.transpose((0, 5, 1, 3, 2, 4)).reshape(B, -1, H, W)
    return out


def _grid_partition(x: Tensor, grid_size: tuple[int, int] = (7, 7)) -> Tensor:
    B, C, H, W = x.shape
    grid = x.reshape(
        B, C, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1]
    )
    grid = grid.transpose((0, 3, 5, 2, 4, 1)).reshape(-1, *grid_size, C)
    return grid


def _grid_reverse(
    grid: Tensor, original_size: tuple[int, int], grid_size: tuple[int, int] = (7, 7)
) -> Tensor:
    NotImplemented  # TODO: Continue from here
