import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


def _make_divisible(
    value: float,
    divisor: int,
    min_value: float | None = None,
    round_down_protect: bool = True,
) -> int:
    if min_value is None:
        min_value = divisor

    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor

    return int(new_value)


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    groups: int = 1,
    bias: bool = False,
    norm: bool = True,
    act: bool = True,
) -> nn.Sequential:
    conv = nn.Sequential()
    conv.add_module(
        "conv",
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding="same",
            bias=bias,
            groups=groups,
        ),
    )
    if norm:
        conv.add_module("bn", nn.BatchNorm2d(out_channels))
    if act:
        conv.add_module("act", nn.ReLU6())

    return conv


class _InvertedResidual(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expand_ratio: int,
        act: bool = False,
        se: bool = False,
    ) -> None:
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride

        hid_channels = int(round(in_channels * expand_ratio))
        self.block = nn.Sequential()

        if expand_ratio != 1:
            self.block.add_module(
                "exp_1x1",
                _make_conv_block(
                    in_channels, hid_channels, kernel_size=3, stride=stride
                ),
            )
        if se:
            self.block.add_module(
                "conv_3x3",
                _make_conv_block(
                    hid_channels,
                    hid_channels,
                    kernel_size=3,
                    stride=stride,
                    groups=hid_channels,
                ),
            )

        self.block.add_module(
            "red_1x1",
            _make_conv_block(
                hid_channels, out_channels, kernel_size=1, stride=1, act=act
            ),
        )
        self.use_residual = self.stride == 1 and in_channels == out_channels

    def forward(self, x: Tensor) -> Tensor:
        if self.use_residual:
            return x + self.block(x)

        return self.block(x)


class _UniversalInvertedBottleneck(nn.Module): ...
