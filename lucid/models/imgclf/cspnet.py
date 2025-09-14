import math
from functools import partial
from typing import Callable, ClassVar, Literal, NamedTuple

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["CSPNet", "csp_resnet_50", "csp_resnext_50_32x4d", "csp_darknet_53"]


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


class _DarknetBottleneck(nn.Module):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = partial(nn.LeakyReLU, negative_slope=0.1),
        **kwargs,
    ) -> None:
        super().__init__()
        self.conv1 = _ConvBNAct(
            in_channels, out_channels, k=1, s=1, p=0, norm=norm, act=act
        )
        self.conv2 = _ConvBNAct(
            out_channels, out_channels, k=3, s=stride, p=1, norm=norm, act=act
        )

        self.use_proj = not (stride == 1 and in_channels == out_channels)
        self.downsample = (
            nn.Identity()
            if not self.use_proj
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                norm(out_channels),
            )
        )
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.use_proj:
            identity = self.downsample(x)

        out += identity
        return self.act(out)


class _StackOut(NamedTuple):
    module: nn.Module
    out_channels: int


def _resnet_stack_factory(
    block_cls: type[nn.Module],
    norm: type[nn.Module] = nn.BatchNorm2d,
    act: type[nn.Module] = nn.ReLU,
    groups: int = 1,
    base_width: int = 64,
    **kwargs,
) -> Callable[[int, int, int], _StackOut]:
    expansion = getattr(block_cls, "expansion", 1)

    def make_stack(
        in_channels: int, num_layers: int, stride_first: int = 1
    ) -> _StackOut:
        channels = math.ceil(in_channels / expansion)
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

    make_stack.required_multiple = expansion
    return make_stack


def _darknet_stack_factory(
    block_cls: type[nn.Module],
    norm: type[nn.Module] = nn.BatchNorm2d,
    act: type[nn.Module] = partial(nn.LeakyReLU, negative_slope=0.1),
    **kwargs,
) -> Callable[[int, int, int], _StackOut]:
    expansion = getattr(block_cls, "expansion", 1)

    def make_stack(
        in_channels: int, num_layers: int, stride_first: int = 1
    ) -> _StackOut:
        channels = in_channels // expansion
        base_kwargs = dict(downsample=None, norm=norm, act=act)

        layers = []
        layers.append(
            block_cls(
                in_channels, channels * expansion, stride=stride_first, **base_kwargs
            )
        )
        in_ch = channels * expansion
        for _ in range(1, num_layers):
            layers.append(
                block_cls(in_ch, channels * expansion, stride=1, **base_kwargs)
            )

        return _StackOut(nn.Sequential(*layers), in_ch)

    make_stack.required_multiple = expansion
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
        pre_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.pre = _ConvBNAct(
            in_channels,
            stage_width,
            k=pre_kernel_size,
            s=(2 if downsample else 1),
            p=(pre_kernel_size - 1) // 2,
            norm=norm,
            act=act,
        )

        c1 = int(round(stage_width * split_ratio))
        c2 = stage_width - c1
        assert c1 > 0 and c2 > 0

        req = getattr(block_stack_fn, "required_multiple", 1)
        if c2 % req != 0:
            c2 = max(req, (c2 // req) * req)
            c1 = stage_width - c2
        self.c1, self.c2 = c1, c2

        self.part1_proj = _ConvBNAct(stage_width, c1, k=1, s=1, p=0, norm=norm, act=act)
        self.part2_proj = _ConvBNAct(stage_width, c2, k=1, s=1, p=0, norm=norm, act=act)

        stack_out = block_stack_fn(c2, num_layers, stride_first=1)
        self.block_stack = stack_out.module
        self.block_out = stack_out.out_channels

        self.merge = _ConvBNAct(
            c1 + self.block_out, stage_width, k=1, s=1, p=0, norm=norm, act=act
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pre(x)
        y1 = self.part1_proj(x)
        y2 = self.part2_proj(x)
        y2 = self.block_stack(y2)

        y = lucid.concatenate([y1, y2], axis=1)
        y = self.merge(y)
        return y


class CSPNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        stem_channels: int = 64,
        stage_defs: list[tuple[int, int, Callable, bool]] | None = None,
        num_classes: int = 1000,
        norm: type[nn.Module] = nn.BatchNorm2d,
        act: type[nn.Module] = nn.ReLU,
        split_ratio: float = 0.5,
        global_pool: Literal["avg", "max"] = "avg",
        dropout: float = 0.0,
        feature_channels: int | None = None,
        pre_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        if stage_defs is None:
            stage_defs = []

        self.stem = nn.Sequential(
            _ConvBNAct(in_channels, stem_channels, k=3, s=2, norm=norm, act=act),
            _ConvBNAct(stem_channels, stem_channels, k=3, s=1, norm=norm, act=act),
        )

        stages = []
        in_ch = stem_channels
        for stage_width, num_layers, stack_fn, downsample in stage_defs:
            stages.append(
                _CSPStage(
                    in_ch,
                    stage_width,
                    num_layers,
                    stack_fn,
                    split_ratio=split_ratio,
                    downsample=downsample,
                    norm=norm,
                    act=act,
                    pre_kernel_size=pre_kernel_size,
                )
            )
            in_ch = stage_width
        self.stages = nn.Sequential(*stages)

        if feature_channels is not None and feature_channels != in_ch:
            self.pre_head = _ConvBNAct(
                in_ch, feature_channels, k=1, s=1, p=0, norm=norm, act=act
            )
            in_ch = feature_channels
        else:
            self.pre_head = nn.Identity()

        self.num_classes = num_classes
        self.head_pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if global_pool == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(in_ch, num_classes),
        )

        self.apply(self.init_weights)

    def init_weights(self, m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight, mode="fan_out")
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant(m.weight, 1.0)
            nn.init.constant(m.bias, 0.0)

    def forward_features(
        self, x: Tensor, return_stage_out: bool = False
    ) -> Tensor | list[Tensor]:
        feats = []
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
            if return_stage_out:
                feats.append(x)

        return feats if return_stage_out else x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x, return_stage_out=False)
        x = self.pre_head(x)
        x = self.head_pool(x)
        x = self.head(x)
        return x


@register_model
def csp_resnet_50(
    num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 64, **kwargs
) -> CSPNet:
    res_stack = _resnet_stack_factory(_Bottleneck, groups=1, base_width=64, **kwargs)
    stage_defs = [
        (256, 3, res_stack, False),
        (512, 4, res_stack, True),
        (1024, 6, res_stack, True),
        (2048, 3, res_stack, True),
    ]
    return CSPNet(
        stem_channels=stem_channels,
        stage_defs=stage_defs,
        num_classes=num_classes,
        split_ratio=split_ratio,
        feature_channels=1024,
        **kwargs,
    )


@register_model
def csp_resnext_50_32x4d(
    num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 64, **kwargs
) -> CSPNet:
    resnext_stack = _resnet_stack_factory(
        _Bottleneck, groups=32, base_width=4, **kwargs
    )
    stage_defs = [
        (256, 3, resnext_stack, False),
        (512, 4, resnext_stack, True),
        (1024, 6, resnext_stack, True),
        (2048, 3, resnext_stack, True),
    ]
    return CSPNet(
        stem_channels=stem_channels,
        stage_defs=stage_defs,
        num_classes=num_classes,
        split_ratio=split_ratio,
        feature_channels=1024,
        **kwargs,
    )


@register_model
def csp_darknet_53(
    num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 32, **kwargs
) -> CSPNet:
    dark_stack = _darknet_stack_factory(_DarknetBottleneck, **kwargs)
    stage_defs = [
        (64, 1, dark_stack, True),
        (128, 2, dark_stack, True),
        (256, 8, dark_stack, True),
        (512, 8, dark_stack, True),
        (1024, 4, dark_stack, True),
    ]
    return CSPNet(
        stem_channels=stem_channels,
        stage_defs=stage_defs,
        num_classes=num_classes,
        split_ratio=split_ratio,
        feature_channels=1024,
        pre_kernel_size=3,
        **kwargs,
    )
