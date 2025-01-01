from typing import ClassVar

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

from .resnet import ResNet


__all__ = [
    "SENet",
    "se_resnet_18",
    "se_resnet_34",
    "se_resnet_50",
    "se_resnet_101",
    "se_resnet_152",
]


class SENet(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        reduction: int = 16,
    ) -> None:
        super().__init__(
            block, layers, num_classes, block_args={"reduction": reduction}
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


class _SEResNetBottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.ConvBNReLU2d(
            in_channels, out_channels, kernel_size=1, conv_bias=False
        )
        self.conv2 = nn.ConvBNReLU2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            conv_bias=False,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels * self.expansion,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels * self.expansion),
        )

        self.se_module = nn.SEModule(out_channels * self.expansion, reduction)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

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
    return SENet(_SEResNetBottleneck, layers, num_classes, **kwargs)


@register_model
def se_resnet_101(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 4, 23, 3]
    return SENet(_SEResNetBottleneck, layers, num_classes, **kwargs)


@register_model
def se_resnet_152(num_classes: int = 1000, **kwargs) -> SENet:
    layers = [3, 8, 36, 3]
    return SENet(_SEResNetBottleneck, layers, num_classes, **kwargs)
