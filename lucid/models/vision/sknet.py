from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

from .resnet import ResNet, ResNetConfig

__all__ = [
    "SKNet",
    "SKNetConfig",
    "sk_resnet_18",
    "sk_resnet_34",
    "sk_resnet_50",
    "sk_resnext_50_32x4d",
]


_SKBlockName = Literal["basic", "bottleneck"]


@dataclass
class SKNetConfig:
    block: _SKBlockName
    layers: tuple[int, int, int, int] | list[int]
    kernel_sizes: tuple[int, ...] | list[int] = (3, 5)
    base_width: int = 64
    cardinality: int = 1
    num_classes: int = 1000
    in_channels: int = 3
    stem_width: int = 64
    stem_type: Literal["deep"] | None = None
    avg_down: bool = False
    channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
    block_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        self.kernel_sizes = tuple(self.kernel_sizes)
        self.channels = tuple(self.channels)
        if not isinstance(self.block_args, dict):
            raise TypeError("block_args must be a dictionary")
        self.block_args = dict(self.block_args)

        if self.block not in ("basic", "bottleneck"):
            raise ValueError("block must be 'basic' or 'bottleneck'")
        if len(self.layers) != 4:
            raise ValueError("layers must contain exactly 4 stage depths")
        if any(not isinstance(depth, int) or depth <= 0 for depth in self.layers):
            raise ValueError("layers values must be positive integers")
        if len(self.kernel_sizes) == 0:
            raise ValueError("kernel_sizes must contain at least one kernel size")
        if any(
            not isinstance(kernel_size, int) or kernel_size <= 0
            for kernel_size in self.kernel_sizes
        ):
            raise ValueError("kernel_sizes values must be positive integers")
        if self.base_width <= 0:
            raise ValueError("base_width must be greater than 0")
        if self.cardinality <= 0:
            raise ValueError("cardinality must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.stem_width <= 0:
            raise ValueError("stem_width must be greater than 0")
        if self.stem_type not in (None, "deep"):
            raise ValueError("stem_type must be None or 'deep'")
        if len(self.channels) != 4:
            raise ValueError("channels must contain exactly 4 stage widths")
        if any(
            not isinstance(channel, int) or channel <= 0 for channel in self.channels
        ):
            raise ValueError("channels values must be positive integers")


def _resolve_sknet_block(block: _SKBlockName) -> type[nn.Module]:
    if block == "basic":
        return _SKResNetModule
    return _SKResNetBottleneck


class SKNet(ResNet):
    def __init__(self, config: SKNetConfig) -> None:
        block_args = {
            **config.block_args,
            "kernel_sizes": list(config.kernel_sizes),
            "base_width": config.base_width,
            "cardinality": config.cardinality,
        }
        super().__init__(
            ResNetConfig(
                block=_resolve_sknet_block(config.block),
                layers=config.layers,
                num_classes=config.num_classes,
                in_channels=config.in_channels,
                stem_width=config.stem_width,
                stem_type=config.stem_type,
                avg_down=config.avg_down,
                channels=config.channels,
                block_args=block_args,
            )
        )
        self.config = config
        self.kernel_sizes = config.kernel_sizes
        self.base_width = config.base_width
        self.cardinality = config.cardinality


class _SKResNetModule(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        kernel_sizes: list[int] = [3, 5],
        cardinality: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * cardinality

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, width, kernel_size=1, stride=1, conv_bias=False
        )
        self.sk_module = nn.SelectiveKernel(
            width, width, kernel_sizes=kernel_sizes, stride=stride, groups=cardinality
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                width,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(self.bn2(self.sk_module(out)))
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _SKResNetBottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        kernel_sizes: list[int] = [3, 5],
        cardinality: int = 1,
        base_width: int = 64,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * cardinality

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, width, kernel_size=1, stride=1, conv_bias=False
        )

        self.sk_module = nn.SelectiveKernel(
            width, width, kernel_sizes=kernel_sizes, stride=stride, groups=cardinality
        )
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                width,
                out_channels * self.expansion,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.relu(self.bn2(self.sk_module(out)))
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def _build_sknet_config(
    *,
    block: _SKBlockName,
    layers: tuple[int, int, int, int] | list[int],
    num_classes: int,
    kwargs: dict[str, Any] | None = None,
    preset_kwargs: dict[str, Any] | None = None,
) -> SKNetConfig:
    kwargs = {} if kwargs is None else dict(kwargs)
    preset_kwargs = {} if preset_kwargs is None else dict(preset_kwargs)
    locked_fields = {"block", "layers"} | set(preset_kwargs)
    if locked_fields & kwargs.keys():
        if preset_kwargs:
            raise TypeError(
                "factory variants do not allow overriding preset block, layers, cardinality, or base_width"
            )
        raise TypeError(
            "factory variants do not allow overriding preset block or layers"
        )

    return SKNetConfig(
        block=block,
        layers=layers,
        num_classes=num_classes,
        **preset_kwargs,
        **kwargs,
    )


@register_model
def sk_resnet_18(num_classes: int = 1000, **kwargs) -> SKNet:
    config = _build_sknet_config(
        block="basic",
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SKNet(config)


@register_model
def sk_resnet_34(num_classes: int = 1000, **kwargs) -> SKNet:
    config = _build_sknet_config(
        block="basic",
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SKNet(config)


@register_model
def sk_resnet_50(num_classes: int = 1000, **kwargs) -> SKNet:
    config = _build_sknet_config(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SKNet(config)


@register_model
def sk_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SKNet:
    config = _build_sknet_config(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        kwargs=kwargs,
        preset_kwargs={"cardinality": 32, "base_width": 4},
    )
    return SKNet(config)
