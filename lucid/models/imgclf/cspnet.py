from typing import Callable, ClassVar, NamedTuple

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


class _ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        groups: int = 1,
        bias: bool = False,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=s,
            padding=p if p is not None else "same",
            groups=groups,
            bias=bias,
        )
        self.norm = norm(out_channels)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class _TransitionPreAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool: bool = False,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.norm = norm(in_channels)
        self.act = act()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(self.act(self.norm(x)))
        x = self.pool(x)
        return x


class _ResNetBasicBlock(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = _ConvBNAct(
            in_channels, out_channels, k=3, s=stride, p=1, norm=norm, act=act
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = norm(out_channels)
        self.act = act()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out


class _Bottleneck(nn.Module):
    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
        groups: int = 1,
        base_width: int = 64,
        **kwargs,
    ) -> None:
        super().__init__()
        width = int(out_channels * (base_width / 64.0)) * groups

        self.conv1 = _ConvBNAct(in_channels, width, k=1, s=1, p=0, norm=norm, act=act)
        self.conv2 = _ConvBNAct(
            width, width, k=3, s=stride, p=1, groups=groups, norm=norm, act=act
        )
        self.conv3 = nn.Conv2d(
            width, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.norm3 = norm(out_channels * self.expansion)
        self.act = act()
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.norm3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)


class _DenseLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        inter = bn_size * growth_rate

        self.norm1 = norm(in_channels)
        self.act1 = act()
        self.conv1 = nn.Conv2d(in_channels, inter, kernel_size=1, bias=False)

        self.norm2 = norm(inter)
        self.act2 = act()
        self.conv2 = nn.Conv2d(inter, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(self.act1(self.norm1(x)))
        out = self.conv2(self.act2(self.norm2(out)))

        if self.drop_rate > 0.0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return lucid.concatenate([x, out], axis=1)


class _DenseBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        growth_rate: int = 32,
        bn_size: int = 4,
        drop_rate: float = 0.0,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        ch = in_channels
        layers = []
        for _ in range(num_layers):
            layer = _DenseLayer(ch, growth_rate, bn_size, drop_rate, norm, act)
            layers.append(layer)
            ch += growth_rate

        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class _StackOut(NamedTuple):
    module: nn.Module
    out_channels: int


def _resnet_stack_factory(
    block_cls: type[nn.Module],
    norm: type[nn.Module] = nn.BatchNorm2d,
    act: type[nn.Module] = nn.ReLU,
    groups: int = 1,
    base_width: int = 64,
) -> Callable[[int, int, int], _StackOut]:
    expansion = getattr(block_cls, "expansion", 1)

    def make_stack(
        in_channels: int, num_layers: int, stride_first: int = 1
    ) -> _StackOut:
        channels = in_channels // expansion
        layers = []

        down = None
        if in_channels != channels * expansion:
            down = nn.Sequential(
                nn.Conv2d(in_channels, channels * expansion, kernel_size=1, bias=False),
                norm(channels * expansion),
            )

        base_kwargs = dict(norm=norm, act=act, groups=groups, base_width=base_width)
        layers.append(
            block_cls(
                in_channels,
                channels,
                stride=stride_first,
                downsample=down,
                **base_kwargs,
            )
        )
        in_channels = channels * expansion
        for _ in range(1, num_layers):
            layers.append(
                block_cls(
                    in_channels, channels, stride=1, downsample=None, **base_kwargs
                )
            )

        return _StackOut(nn.Sequential(*layers), in_channels)

    return make_stack


def densenet_stack_factory(
    growth_rate: int = 32,
    bn_size: int = 4,
    drop_rate: float = 0.0,
    norm: type[nn.Module] = nn.BatchNorm2d,
    act: type[nn.Module] = nn.ReLU,
) -> Callable[[int, int, int], _StackOut]:

    def make_stack(
        in_channels: int, num_layers: int, stride_first: int = 1
    ) -> _StackOut:
        _ = stride_first
        block = _DenseBlock(
            in_channels, num_layers, growth_rate, bn_size, drop_rate, norm, act
        )
        return _StackOut(block, block.out_channels)

    return make_stack


class _CSPStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stage_width: int,
        num_layers: int,
        block_stack_fn: Callable[[int, int, int], _StackOut],
        split_ratio: float = 0.5,
        downsample: bool = False,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.stage_width = stage_width
        self.split_c1 = int(stage_width * split_ratio)
        self.split_c2 = stage_width - self.split_c1
        assert self.split_c1 > 0 and self.split_c2 > 0

        self.base_align = (
            nn.Identity()
            if in_channels == stage_width
            else _ConvBNAct(in_channels, stage_width, k=1, s=1, p=0, norm=norm, act=act)
        )

        # TODO: Continue from here
