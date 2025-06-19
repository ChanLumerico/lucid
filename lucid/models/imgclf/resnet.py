from typing import Any, ClassVar, Type, Literal
import math

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "ResNet",
    "resnet_18",
    "resnet_34",
    "resnet_50",
    "resnet_101",
    "resnet_152",
    "resnet_200",
    "resnet_269",
    "resnet_1001",
    "wide_resnet_50",
    "wide_resnet_101",
]


class ResNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        stem_width: int = 64,
        stem_type: Literal["deep"] | None = None,
        avg_down: bool = False,
        channels: tuple[int] = (64, 128, 256, 512),
        block_args: dict[str, Any] = {},
    ) -> None:
        super().__init__()
        deep_stem = stem_type == "deep"
        self.in_channels = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down

        if deep_stem:
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(),
                nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(stem_width),
                nn.ReLU(),
                nn.Conv2d(stem_width, self.in_channels, 3, padding=1, bias=False),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2d(
                    in_channels, self.in_channels, 7, stride=2, padding=3, bias=False
                ),
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
            )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, channels[0], layers[0], stride=1, block_args=block_args
        )
        self.layer2 = self._make_layer(
            block, channels[1], layers[1], stride=2, block_args=block_args
        )
        self.layer3 = self._make_layer(
            block, channels[2], layers[2], stride=2, block_args=block_args
        )
        self.layer4 = self._make_layer(
            block, channels[3], layers[3], stride=2, block_args=block_args
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int = 1,
        block_args: dict[str, Any] = {},
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            if self.avg_down:
                downsample = nn.Sequential(
                    nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                    nn.Conv2d(
                        self.in_channels,
                        out_channels * block.expansion,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.in_channels,
                        out_channels * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels * block.expansion),
                )

        layers = [
            block(self.in_channels, out_channels, stride, downsample, **block_args)
        ]
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1, **block_args))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.maxpool(x)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            x = layer(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class _BasicBlock(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        basic_conv_args = dict(kernel_size=3, padding=1, bias=False)

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, stride=stride, **basic_conv_args
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, stride=1, **basic_conv_args)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class _Bottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        cardinality: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        se: bool = False,
        se_args: dict = {},
    ) -> None:
        super().__init__()
        width = int(math.floor(out_channels * (base_width / 64)) * cardinality)

        self.conv1 = nn.ConvBNReLU2d(
            in_channels, width, kernel_size=1, stride=1, conv_bias=False
        )
        self.conv2 = nn.ConvBNReLU2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            dilation=dilation,
            groups=cardinality,
            conv_bias=False,
        )
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

        self.se = nn.SEModule(out_channels * self.expansion, **se_args) if se else None
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _PreActBottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )

        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu1(self.bn1(x))
        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)
        out = self.conv2(self.relu2(self.bn2(out)))
        out = self.conv3(self.relu3(self.bn3(out)))

        out += identity
        return out


@register_model
def resnet_18(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [2, 2, 2, 2]
    return ResNet(_BasicBlock, layers, num_classes, **kwargs)


@register_model
def resnet_34(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 6, 3]
    return ResNet(_BasicBlock, layers, num_classes, **kwargs)


@register_model
def resnet_50(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 6, 3]
    return ResNet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def resnet_101(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 23, 3]
    return ResNet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def resnet_152(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 8, 36, 3]
    return ResNet(_Bottleneck, layers, num_classes, **kwargs)


@register_model
def resnet_200(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 24, 36, 3]
    return ResNet(_PreActBottleneck, layers, num_classes, **kwargs)


@register_model
def resnet_269(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 30, 48, 8]
    return ResNet(_PreActBottleneck, layers, num_classes, **kwargs)


@register_model
def resnet_1001(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 94, 94, 3]
    return ResNet(_PreActBottleneck, layers, num_classes, **kwargs)


@register_model
def wide_resnet_50(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 6, 3]
    block_args = {"base_width": 128}
    return ResNet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)


@register_model
def wide_resnet_101(num_classes: int = 1000, **kwargs) -> ResNet:
    layers = [3, 4, 23, 3]
    block_args = {"base_width": 128}
    return ResNet(_Bottleneck, layers, num_classes, block_args=block_args, **kwargs)
