from dataclasses import dataclass

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor
from lucid_legacy.models.base import PreTrainedModelMixin

__all__ = [
    "DenseNet",
    "DenseNetConfig",
    "densenet_121",
    "densenet_169",
    "densenet_201",
    "densenet_264",
]


@dataclass
class DenseNetConfig:
    block_config: tuple[int, int, int, int] | list[int]
    growth_rate: int = 32
    num_init_features: int = 64
    num_classes: int = 1000
    in_channels: int = 3
    bottleneck: int = 4
    compression: float = 0.5

    def __post_init__(self) -> None:
        self.block_config = tuple(self.block_config)

        if len(self.block_config) != 4:
            raise ValueError("block_config must contain exactly 4 dense block depths")
        if any(not isinstance(depth, int) or depth <= 0 for depth in self.block_config):
            raise ValueError("block_config values must be positive integers")
        if self.growth_rate <= 0:
            raise ValueError("growth_rate must be greater than 0")
        if self.num_init_features <= 0:
            raise ValueError("num_init_features must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.bottleneck <= 0:
            raise ValueError("bottleneck must be greater than 0")
        if self.compression <= 0 or self.compression > 1:
            raise ValueError("compression must be in the range (0, 1]")


class DenseNet(nn.Module, PreTrainedModelMixin):
    def __init__(self, config: DenseNetConfig) -> None:
        super().__init__()
        self.config = config
        self.growth_rate = config.growth_rate
        self.num_init_features = config.num_init_features
        self.bottleneck = config.bottleneck
        self.compression = config.compression

        self.conv0 = nn.ConvBNReLU2d(
            config.in_channels,
            config.num_init_features,
            kernel_size=7,
            stride=2,
            padding=3,
            conv_bias=False,
        )
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        in_channels = config.num_init_features
        for i, num_layers in enumerate(config.block_config):
            block = _DenseBlock(
                in_channels,
                num_layers,
                config.growth_rate,
                bottleneck=config.bottleneck,
            )
            self.blocks.append(block)

            in_channels += num_layers * config.growth_rate

            if i != len(config.block_config) - 1:
                out_channels = int(in_channels * config.compression)
                transition = _TransitionLayer(in_channels, out_channels)
                self.transitions.append(transition)

                in_channels = out_channels

        self.bn_final = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, config.num_classes)

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


def _build_densenet_config(
    *,
    block_config: tuple[int, int, int, int] | list[int],
    growth_rate: int,
    num_init_features: int,
    num_classes: int,
    kwargs: dict[str, object] | None = None,
) -> DenseNetConfig:
    kwargs = {} if kwargs is None else dict(kwargs)
    locked_fields = {"block_config", "growth_rate", "num_init_features"}
    if locked_fields & kwargs.keys():
        raise TypeError(
            "factory variants do not allow overriding preset block_config, growth_rate, or num_init_features"
        )

    return DenseNetConfig(
        block_config=block_config,
        growth_rate=growth_rate,
        num_init_features=num_init_features,
        num_classes=num_classes,
        **kwargs,
    )


@register_model
def densenet_121(num_classes: int = 1000, **kwargs) -> DenseNet:
    config = _build_densenet_config(
        block_config=(6, 12, 24, 16),
        growth_rate=32,
        num_init_features=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return DenseNet(config)


@register_model
def densenet_169(num_classes: int = 1000, **kwargs) -> DenseNet:
    config = _build_densenet_config(
        block_config=(6, 12, 32, 32),
        growth_rate=32,
        num_init_features=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return DenseNet(config)


@register_model
def densenet_201(num_classes: int = 1000, **kwargs) -> DenseNet:
    config = _build_densenet_config(
        block_config=(6, 12, 48, 32),
        growth_rate=32,
        num_init_features=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return DenseNet(config)


@register_model
def densenet_264(num_classes: int = 1000, **kwargs) -> DenseNet:
    config = _build_densenet_config(
        block_config=(6, 12, 64, 48),
        growth_rate=32,
        num_init_features=64,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return DenseNet(config)
