from typing import Any, ClassVar
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

from .resnet import ResNet


__all__ = [
    "SKNet",
    "sk_resnet_18",
    "sk_resnet_34",
    "sk_resnet_50",
    "sk_resnext_50_32x4d",
]


class SKNet(ResNet):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        kernel_sizes: list[int] = [3, 5],
        base_width: int = 64,
        cardinality: int = 1,
        **resnet_args: Any,
    ) -> None:
        super().__init__(
            block,
            layers,
            num_classes,
            block_args={
                "kernel_sizes": kernel_sizes,
                "base_width": base_width,
                "cardinality": cardinality,
            },
            **resnet_args,
        )


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


@register_model
def sk_resnet_18(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [2, 2, 2, 2]
    return SKNet(_SKResNetModule, layers, num_classes, **kwargs)


@register_model
def sk_resnet_34(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [3, 4, 6, 3]
    return SKNet(_SKResNetModule, layers, num_classes, **kwargs)


@register_model
def sk_resnet_50(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [3, 4, 6, 3]
    return SKNet(_SKResNetBottleneck, layers, num_classes, **kwargs)


@register_model
def sk_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SKNet:
    layers = [3, 4, 6, 3]
    return SKNet(
        _SKResNetBottleneck,
        layers,
        num_classes,
        cardinality=32,
        base_width=4,
        **kwargs,
    )
