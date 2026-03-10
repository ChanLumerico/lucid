from dataclasses import dataclass

import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

__all__ = ["Xception", "XceptionConfig", "xception"]


@dataclass
class XceptionConfig:
    num_classes: int = 1000
    in_channels: int = 3
    stem_channels: tuple[int, int] | list[int] = (32, 64)
    entry_channels: tuple[int, int, int] | list[int] = (128, 256, 728)
    middle_channels: int = 728
    middle_repeats: int = 8
    exit_channels: tuple[int, int, int] | list[int] = (1024, 1536, 2048)

    def __post_init__(self) -> None:
        self.stem_channels = tuple(self.stem_channels)
        self.entry_channels = tuple(self.entry_channels)
        self.exit_channels = tuple(self.exit_channels)

        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if len(self.stem_channels) != 2:
            raise ValueError("stem_channels must contain exactly 2 channel values")
        if any(
            not isinstance(channel, int) or channel <= 0
            for channel in self.stem_channels
        ):
            raise ValueError("stem_channels values must be positive integers")
        if len(self.entry_channels) != 3:
            raise ValueError("entry_channels must contain exactly 3 channel values")
        if any(
            not isinstance(channel, int) or channel <= 0
            for channel in self.entry_channels
        ):
            raise ValueError("entry_channels values must be positive integers")
        if self.middle_channels <= 0:
            raise ValueError("middle_channels must be greater than 0")
        if self.middle_repeats <= 0:
            raise ValueError("middle_repeats must be greater than 0")
        if len(self.exit_channels) != 3:
            raise ValueError("exit_channels must contain exactly 3 channel values")
        if any(
            not isinstance(channel, int) or channel <= 0
            for channel in self.exit_channels
        ):
            raise ValueError("exit_channels values must be positive integers")


class _Block(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reps: int,
        stride: int = 1,
        start_with_relu: bool = True,
        grow_first: bool = True,
    ) -> None:
        super().__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.skipbn = nn.BatchNorm2d(out_channels)
        else:
            self.skip = None

        rep = []
        channels = in_channels
        if grow_first:
            rep.append(nn.ReLU())
            rep.append(
                nn.DepthSeparableConv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_channels))
            channels = out_channels

        for _ in range(reps - 1):
            rep.append(nn.ReLU())
            rep.append(
                nn.DepthSeparableConv2d(
                    channels, channels, kernel_size=3, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(channels))

        if not grow_first:
            rep.append(nn.ReLU())
            rep.append(
                nn.DepthSeparableConv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                )
            )
            rep.append(nn.BatchNorm2d(out_channels))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU()

        if stride != 1:
            rep.append(nn.MaxPool2d(kernel_size=3, stride=stride, padding=1))

        self.rep = nn.Sequential(*rep)

    def forward(self, x: Tensor) -> Tensor:
        out = self.rep(x)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x

        out += skip
        return out


class Xception(nn.Module):
    def __init__(self, config: XceptionConfig) -> None:
        super().__init__()
        self.config = config
        self.relu = nn.ReLU()

        self.conv1 = nn.ConvBNReLU2d(
            config.in_channels,
            config.stem_channels[0],
            kernel_size=3,
            stride=2,
            padding=0,
            conv_bias=False,
        )
        self.conv2 = nn.ConvBNReLU2d(
            config.stem_channels[0],
            config.stem_channels[1],
            kernel_size=3,
            conv_bias=False,
        )

        self.block1 = _Block(
            config.stem_channels[1],
            config.entry_channels[0],
            reps=2,
            stride=2,
            start_with_relu=False,
        )
        self.block2 = _Block(
            config.entry_channels[0], config.entry_channels[1], reps=2, stride=2
        )
        self.block3 = _Block(
            config.entry_channels[1], config.entry_channels[2], reps=2, stride=2
        )

        self.mid_blocks = nn.Sequential(
            *[
                _Block(config.middle_channels, config.middle_channels, reps=3)
                for _ in range(config.middle_repeats)
            ]
        )
        self.end_block = _Block(
            config.middle_channels,
            config.exit_channels[0],
            reps=2,
            stride=2,
            grow_first=False,
        )

        self.conv3 = nn.DepthSeparableConv2d(
            config.exit_channels[0],
            config.exit_channels[1],
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(config.exit_channels[1])

        self.conv4 = nn.DepthSeparableConv2d(
            config.exit_channels[1],
            config.exit_channels[2],
            kernel_size=3,
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(config.exit_channels[2])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(config.exit_channels[2], config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.block3(self.block2(self.block1(x)))
        x = self.mid_blocks(x)
        x = self.end_block(x)

        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def _build_xception_config(
    *,
    num_classes: int,
    kwargs: dict[str, object] | None = None,
) -> XceptionConfig:
    kwargs = {} if kwargs is None else dict(kwargs)
    locked_fields = {
        "stem_channels",
        "entry_channels",
        "middle_channels",
        "middle_repeats",
        "exit_channels",
    }
    if locked_fields & kwargs.keys():
        raise TypeError(
            "factory variants do not allow overriding preset stem_channels, entry_channels, middle_channels, middle_repeats, or exit_channels"
        )

    return XceptionConfig(num_classes=num_classes, **kwargs)


@register_model
def xception(num_classes: int = 1000, **kwargs) -> Xception:
    config = _build_xception_config(num_classes=num_classes, kwargs=kwargs)
    return Xception(config)
