from typing import ClassVar

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

from .resnet import ResNet, _Bottleneck


__all__ = [
    "SENet",
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


class SENet(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        reduction: int = 16,
        block_args: dict = {},
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            block_args={"se": True, "se_args": dict(reduction=reduction), **block_args},
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
    return SENet(_SEResNetModule, layers, num_classes, **kwargs)


@register_model
def se_resnet_34(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    return SENet(_SEResNetModule, layers, num_classes, **kwargs)


@register_model
def se_resnet_50(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    return SENet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def se_resnet_101(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    return SENet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def se_resnet_152(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 8, 36, 3]
    return SENet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def se_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 6, 3]
    block_args = {"cardinality": 32, "base_width": 4}
    return SENet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)


@register_model
def se_resnext_101_32x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    block_args = {"cardinality": 32, "base_width": 4}
    return SENet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)


@register_model
def se_resnext_101_32x8d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    block_args = {"cardinality": 32, "base_width": 8}
    return SENet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)


@register_model
def se_resnext_101_64x4d(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    block_args = {"cardinality": 64, "base_width": 4}
    return SENet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)
