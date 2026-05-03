from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

from .resnet import ResNet, ResNetConfig, _Bottleneck

__all__ = [
    "SENet",
    "SENetConfig",
    "se_resnet_18",
    "se_resnet_34",
    "se_resnet_50",
    "se_resnet_101",
    "se_resnet_152",
    "se_resnext_50_32x4d",
    "se_resnext_101_32x4d",
    "se_resnext_101_32x8d",
    "se_resnext_101_64x4d",
]


@dataclass
class SENetConfig:
    block: Literal["se_basic", "bottleneck"]
    layers: tuple[int, int, int, int] | list[int]
    reduction: int = 16
    cardinality: int = 1
    base_width: int = 64
    num_classes: int = 1000
    in_channels: int = 3
    stem_width: int = 64
    stem_type: Literal["deep"] | None = None
    avg_down: bool = False
    channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
    block_args: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.layers = tuple(self.layers)
        self.channels = tuple(self.channels)
        if not isinstance(self.block_args, dict):
            raise TypeError("block_args must be a dictionary")
        self.block_args = dict(self.block_args)

        if self.block not in ("se_basic", "bottleneck"):
            raise ValueError("block must be 'se_basic' or 'bottleneck'")
        if len(self.layers) != 4:
            raise ValueError("layers must contain exactly 4 stage depths")
        if any(not isinstance(depth, int) or depth <= 0 for depth in self.layers):
            raise ValueError("layers values must be positive integers")
        if self.reduction <= 0:
            raise ValueError("reduction must be greater than 0")
        if self.cardinality <= 0:
            raise ValueError("cardinality must be greater than 0")
        if self.base_width <= 0:
            raise ValueError("base_width must be greater than 0")
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


class SENet(ResNet):
    def __init__(self, config: SENetConfig) -> None:
        block = _SEResNetModule if config.block == "se_basic" else _Bottleneck
        if config.block == "se_basic":
            block_args = {
                **config.block_args,
                "reduction": config.reduction,
            }
        else:
            block_args = {
                **config.block_args,
                "se": True,
                "se_args": dict(reduction=config.reduction),
                "cardinality": config.cardinality,
                "base_width": config.base_width,
            }
        super().__init__(
            ResNetConfig(
                block=block,
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
        self.reduction = config.reduction
        self.cardinality = config.cardinality
        self.base_width = config.base_width


def _build_senet_config(
    *,
    block: Literal["se_basic", "bottleneck"],
    layers: tuple[int, int, int, int] | list[int],
    reduction: int,
    cardinality: int,
    base_width: int,
    num_classes: int,
    kwargs: dict[str, Any] | None = None,
) -> SENetConfig:
    kwargs = {} if kwargs is None else dict(kwargs)
    locked_fields = {"block", "layers", "reduction", "cardinality", "base_width"}
    if locked_fields & kwargs.keys():
        raise TypeError(
            "factory variants do not allow overriding preset block, layers, reduction, cardinality, or base_width"
        )

    return SENetConfig(
        block=block,
        layers=layers,
        reduction=reduction,
        cardinality=cardinality,
        base_width=base_width,
        num_classes=num_classes,
        **kwargs,
    )


class _SEResNetModule(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()

        self.conv1 = nn.ConvBNReLU2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_bias=False,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        self.se_module = nn.SEModule(out_channels, reduction)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)

        out = self.se_module(out)
        if self.downsample is not None:
            out += self.downsample(x)

        out = self.relu(out)
        return out


@register_model
def se_resnet_18(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [2, 2, 2, 2]
    config = _build_senet_config(
        block="se_basic",
        layers=layers,
        reduction=16,
        cardinality=1,
        base_width=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnet_34(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    config = _build_senet_config(
        block="se_basic",
        layers=layers,
        reduction=16,
        cardinality=1,
        base_width=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnet_50(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    config = _build_senet_config(
        block="bottleneck",
        layers=layers,
        reduction=16,
        cardinality=1,
        base_width=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnet_101(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    config = _build_senet_config(
        block="bottleneck",
        layers=layers,
        reduction=16,
        cardinality=1,
        base_width=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnet_152(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 8, 36, 3]
    config = _build_senet_config(
        block="bottleneck",
        layers=layers,
        reduction=16,
        cardinality=1,
        base_width=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    config = _build_senet_config(
        block="bottleneck",
        layers=layers,
        reduction=16,
        cardinality=32,
        base_width=4,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnext_101_32x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    config = _build_senet_config(
        block="bottleneck",
        layers=layers,
        reduction=16,
        cardinality=32,
        base_width=4,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnext_101_32x8d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    config = _build_senet_config(
        block="bottleneck",
        layers=layers,
        reduction=16,
        cardinality=32,
        base_width=8,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)


@register_model
def se_resnext_101_64x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    config = _build_senet_config(
        block="bottleneck",
        layers=layers,
        reduction=16,
        cardinality=64,
        base_width=4,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return SENet(config)
