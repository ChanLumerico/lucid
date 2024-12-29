from typing import ClassVar, Type
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "SENet",
    "se_resnet_18",
    "se_resnet_34",
    "se_resnet_50",
    "se_resnet_101",
    "se_resnet_152",
]


class SENet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = 64

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = self._make_layer(
            block, 64, layers[0], stride=1, reduction=reduction
        )
        self.stage2 = self._make_layer(
            block, 128, layers[1], stride=2, reduction=reduction
        )
        self.stage3 = self._make_layer(
            block, 256, layers[2], stride=2, reduction=reduction
        )
        self.stage4 = self._make_layer(
            block, 512, layers[3], stride=2, reduction=reduction
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int,
        reduction: int,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
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

        layers = [block(self.in_channels, out_channels, stride, reduction, downsample)]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, 1, reduction))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class _SEModule(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.avgpool(x).squeeze(axis=(-2, -1))
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))

        y = y.unsqueeze(axis=(-2, -1))
        out = x * y
        return out


class _SEResNetModule(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
        downsample: nn.Module | None = None,
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

        self.se_module = _SEModule(out_channels, reduction)
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
        reduction: int = 16,
        downsample: nn.Module | None = None,
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

        self.se_module = _SEModule(out_channels * self.expansion, reduction)
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
