from dataclasses import dataclass, field
from typing import Any, ClassVar

import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

from .resnet import ResNet, ResNetConfig

__all__ = [
    "ResNeSt",
    "ResNeStConfig",
    "resnest_14",
    "resnest_26",
    "resnest_50",
    "resnest_101",
    "resnest_200",
    "resnest_269",
    "resnest_50_4s2x40d",
    "resnest_50_1s4x24d",
]


@dataclass
class ResNeStConfig:
    layers: tuple[int, int, int, int] | list[int]
    base_width: int = 64
    stem_width: int = 32
    cardinality: int = 1
    radix: int = 2
    avd: bool = True
    num_classes: int = 1000
    in_channels: int = 3
    avg_down: bool = False
    channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
    block_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        self.channels = tuple(self.channels)
        if not isinstance(self.block_args, dict):
            raise TypeError("block_args must be a dictionary")
        self.block_args = dict(self.block_args)

        if len(self.layers) != 4:
            raise ValueError("layers must contain exactly 4 stage depths")
        if any(not isinstance(depth, int) or depth <= 0 for depth in self.layers):
            raise ValueError("layers values must be positive integers")
        if self.base_width <= 0:
            raise ValueError("base_width must be greater than 0")
        if self.stem_width <= 0:
            raise ValueError("stem_width must be greater than 0")
        if self.cardinality <= 0:
            raise ValueError("cardinality must be greater than 0")
        if self.radix < 0:
            raise ValueError("radix must be greater than or equal to 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if len(self.channels) != 4:
            raise ValueError("channels must contain exactly 4 stage widths")
        if any(
            not isinstance(channel, int) or channel <= 0 for channel in self.channels
        ):
            raise ValueError("channels values must be positive integers")


class _RadixSoftmax(nn.Module):
    def __init__(self, radix: int, cardinality: int) -> None:
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        if self.radix > 1:
            x = x.reshape(N, self.radix, self.cardinality, -1)
            x = F.softmax(x, axis=1)
            x = x.reshape(N, -1)
        else:
            x = F.sigmoid(x)

        return x


class _SplitAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int | str = 0,
        groups: int = 1,
        bias: bool = False,
        radix: int = 2,
        reduce: int = 4,
        drop_layer: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        mid_channels = out_channels * radix
        attn_channels = max(32, int(in_channels * radix / reduce))

        self.conv = nn.Conv2d(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            padding,
            groups=groups * radix,
            bias=bias,
        )
        self.bn0 = nn.BatchNorm2d(mid_channels)
        self.drop = drop_layer() if drop_layer is not None else nn.Identity()
        self.relu0 = nn.ReLU()

        self.fc1 = nn.ConvBNReLU2d(
            out_channels, attn_channels, kernel_size=1, groups=groups
        )
        self.fc2 = nn.Conv2d(attn_channels, mid_channels, kernel_size=1, groups=groups)
        self.rsoftmax = _RadixSoftmax(radix, cardinality=groups)
        self.radix = radix

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn0(x)
        x = self.drop(x)
        x = self.relu0(x)

        N, C, H, W = x.shape
        if self.radix > 1:
            x = x.reshape(N, self.radix, C // self.radix, H, W)
            x_gap = x.sum(axis=1)
        else:
            x_gap = x

        x_gap = x_gap.mean(axis=(2, 3), keepdims=True)
        x_gap = self.fc1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).reshape(N, -1, 1, 1)

        if self.radix > 1:
            out = (x * x_attn.reshape(N, self.radix, C // self.radix, 1, 1)).sum(axis=1)
        else:
            out = x * x_attn
        return out


class _ResNeStBottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        radix: int = 1,
        cardinality: int = 1,
        base_width: int = 64,
        avd: bool = False,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        group_width = int(out_channels * (base_width / 64.0)) * cardinality
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            stride = 1
        else:
            avd_stride = 0

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, group_width, kernel_size=1, conv_bias=False
        )

        conv2_args = (group_width, group_width, 3, stride, 1)
        if radix >= 1:
            self.conv2 = _SplitAttention(*conv2_args, groups=cardinality, radix=radix)
        else:
            self.conv2 = nn.ConvBNReLU2d(
                *conv2_args, groups=cardinality, conv_bias=False
            )

        self.avd_last = (
            nn.AvgPool2d(kernel_size=3, stride=avd_stride, padding=1)
            if avd_stride > 0
            else None
        )
        self.conv3 = nn.Conv2d(
            group_width, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.avd_last is not None:
            out = self.avd_last(out)

        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)
        return out


class ResNeSt(ResNet):
    def __init__(self, config: ResNeStConfig) -> None:
        block_args = {
            **config.block_args,
            "base_width": config.base_width,
            "cardinality": config.cardinality,
            "radix": config.radix,
            "avd": config.avd,
        }
        super().__init__(
            ResNetConfig(
                block=_ResNeStBottleneck,
                layers=config.layers,
                num_classes=config.num_classes,
                in_channels=config.in_channels,
                stem_width=config.stem_width,
                stem_type="deep",
                avg_down=config.avg_down,
                channels=config.channels,
                block_args=block_args,
            )
        )
        self.config = config
        self.base_width = config.base_width
        self.stem_width = config.stem_width
        self.cardinality = config.cardinality
        self.radix = config.radix
        self.avd = config.avd


def _build_resnest_config(
    *,
    layers: tuple[int, int, int, int] | list[int],
    base_width: int,
    stem_width: int,
    cardinality: int,
    radix: int,
    avd: bool,
    num_classes: int,
    kwargs: dict[str, Any] | None = None,
) -> ResNeStConfig:
    kwargs = {} if kwargs is None else dict(kwargs)
    locked_fields = {
        "layers",
        "base_width",
        "stem_width",
        "cardinality",
        "radix",
        "avd",
    }
    if locked_fields & kwargs.keys():
        raise TypeError(
            "factory variants do not allow overriding preset layers, base_width, "
            "stem_width, cardinality, radix, or avd"
        )

    return ResNeStConfig(
        layers=layers,
        base_width=base_width,
        stem_width=stem_width,
        cardinality=cardinality,
        radix=radix,
        avd=avd,
        num_classes=num_classes,
        **kwargs,
    )


@register_model
def resnest_14(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [1, 1, 1, 1]
    config = _build_resnest_config(
        layers=layers,
        base_width=64,
        stem_width=32,
        cardinality=1,
        radix=2,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)


@register_model
def resnest_26(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [2, 2, 2, 2]
    config = _build_resnest_config(
        layers=layers,
        base_width=64,
        stem_width=32,
        cardinality=1,
        radix=2,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)


@register_model
def resnest_50(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [3, 4, 6, 3]
    config = _build_resnest_config(
        layers=layers,
        base_width=64,
        stem_width=32,
        cardinality=1,
        radix=2,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)


@register_model
def resnest_101(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [3, 4, 23, 3]
    config = _build_resnest_config(
        layers=layers,
        base_width=64,
        stem_width=64,
        cardinality=1,
        radix=2,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)


@register_model
def resnest_200(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [3, 24, 36, 3]
    config = _build_resnest_config(
        layers=layers,
        base_width=64,
        stem_width=64,
        cardinality=1,
        radix=2,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)


@register_model
def resnest_269(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [3, 30, 48, 8]
    config = _build_resnest_config(
        layers=layers,
        base_width=64,
        stem_width=64,
        cardinality=1,
        radix=2,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)


@register_model
def resnest_50_4s2x40d(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [3, 4, 6, 3]
    config = _build_resnest_config(
        layers=layers,
        base_width=40,
        stem_width=32,
        cardinality=2,
        radix=4,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)


@register_model
def resnest_50_1s4x24d(num_classes: int = 1000, **kwargs) -> ResNeSt:
    layers = [3, 4, 6, 3]
    config = _build_resnest_config(
        layers=layers,
        base_width=24,
        stem_width=32,
        cardinality=4,
        radix=1,
        avd=True,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return ResNeSt(config)
