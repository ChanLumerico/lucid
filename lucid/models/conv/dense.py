import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "DenseNet",
    "densenet_121",
    "densenet_169",
    "densenet_201",
    "densenet_264",
]


class DenseNet(nn.Module):
    def __init__(
        self,
        block_config: tuple[int],
        growth_rate: int = 32,
        num_init_features: int = 64,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.conv0 = nn.ConvBNReLU2d(
            3, num_init_features, kernel_size=7, stride=2, padding=3, conv_bias=False
        )
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        in_channels = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(in_channels, num_layers, growth_rate)
            self.blocks.append(block)

            in_channels += num_layers * growth_rate

            if i != len(block_config) - 1:
                out_channels = in_channels // 2
                transition = _TransitionLayer(in_channels, out_channels)
                self.transitions.append(transition)

                in_channels = out_channels

        self.bn_final = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.pool0(self.conv0(x))

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)

        x = self.avgpool(self.relu(self.bn_final(x)))
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class _DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bottleneck: int = 4,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels, bottleneck * growth_rate, kernel_size=1, bias=False
        )

        self.bn2 = nn.BatchNorm2d(bottleneck * growth_rate)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            bottleneck * growth_rate,
            growth_rate,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))

        return lucid.concatenate([x, out], axis=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int,
        bottleneck: int = 4,
    ) -> None:
        super().__init__()
        layers = [
            _DenseLayer(in_channels + i * growth_rate, growth_rate, bottleneck)
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class _TransitionLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


@register_model
def densenet_121(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 24, 16)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)


@register_model
def densenet_169(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 32, 32)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)


@register_model
def densenet_201(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 48, 32)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)


@register_model
def densenet_264(num_classes: int = 1000, **kwargs) -> DenseNet:
    block_config = (6, 12, 64, 48)
    return DenseNet(block_config, num_classes=num_classes, **kwargs)
