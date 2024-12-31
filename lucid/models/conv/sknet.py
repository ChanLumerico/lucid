from typing import ClassVar, Type, Literal
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["SKNet", "sk_resnet_18", "sk_resnet_34", "sk_resnet_50"]


class SKNet(nn.Module):
    def __init__(
        self,
        block: nn.Module,
        layers: list[int],
        num_classes: int = 1000,
        kernel_sizes: list[int] = [3, 5],
        cardinality: int = 1,
        base_width: int = 64,
        stem_width: int = 64,
        stem_type: Literal["deep"] | None = None,
    ) -> None:
        super().__init__()
        deep_stem: bool = stem_type == "deep"
        self.in_channels = stem_width * 2 if deep_stem else 64

        stem_conv = None
        if deep_stem:  # TODO: Need to fix channel issue here
            stem_conv = nn.Sequential(
                nn.ConvBNReLU2d(
                    3, stem_width, kernel_size=3, stride=2, padding=1, conv_bias=False
                ),
                nn.ConvBNReLU2d(
                    stem_width, stem_width, kernel_size=3, padding=1, conv_bias=False
                ),
                nn.ConvBNReLU2d(
                    stem_width,
                    self.in_channels,
                    kernel_size=3,
                    padding=1,
                    conv_bias=False,
                ),
            )

        self.conv = nn.Sequential(
            stem_conv, nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.stage1 = self._make_layer(
            block, 64, layers[0], 1, kernel_sizes, cardinality, base_width
        )
        self.stage2 = self._make_layer(
            block, 128, layers[1], 2, kernel_sizes, cardinality, base_width
        )
        self.stage3 = self._make_layer(
            block, 256, layers[2], 2, kernel_sizes, cardinality, base_width
        )
        self.stage4 = self._make_layer(
            block, 512, layers[3], 2, kernel_sizes, cardinality, base_width
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[nn.Module],
        out_channels: int,
        blocks: int,
        stride: int,
        kernel_sizes: list[int],
        cardinality: int,
        base_width: int,
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

        layers = [
            block(
                self.in_channels,
                out_channels,
                kernel_sizes=kernel_sizes,
                stride=stride,
                cardinality=cardinality,
                downsample=downsample,
                base_width=base_width,
            )
        ]
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    kernel_sizes=kernel_sizes,
                    stride=1,
                    cardinality=cardinality,
                    base_width=base_width,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)

        for stage in [self.stage1, self.stage2, self.stage3, self.stage4]:
            x = stage(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class _SKResNetModule(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_sizes: list[int],
        stride: int = 1,
        downsample: nn.Module | None = None,
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
        kernel_sizes: list[int],
        stride: int = 1,
        downsample: nn.Module | None = None,
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
