import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar, Literal, NamedTuple

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "CSPNet",
    "CSPNetConfig",
    "csp_resnet_50",
    "csp_resnext_50_32x4d",
    "csp_darknet_53",
]


def _normalize_stage_specs(
    values: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | list[list[object]],
) -> tuple[tuple[int, int, bool], ...]:
    normalized = tuple(tuple(spec) for spec in values)
    if len(normalized) == 0:
        raise ValueError("stage_specs must contain at least one stage spec")
    for spec in normalized:
        if len(spec) != 3:
            raise ValueError(
                "each stage spec must contain exactly 3 values: (stage_width, num_layers, downsample)"
            )
        stage_width, num_layers, downsample = spec
        if not isinstance(stage_width, int) or stage_width <= 0:
            raise ValueError("stage_width values must be positive integers")
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError("num_layers values must be positive integers")
        if not isinstance(downsample, bool):
            raise TypeError("downsample values must be booleans")
    return normalized  # type: ignore[return-value]


@dataclass
class CSPNetConfig:
    stage_specs: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | list[list[object]]
    stack_type: Literal["resnet", "resnext", "darknet"]
    in_channels: int = 3
    stem_channels: int = 64
    num_classes: int = 1000
    norm: Callable[..., nn.Module] = nn.BatchNorm2d
    act: Callable[..., nn.Module] = nn.ReLU
    split_ratio: float = 0.5
    global_pool: Literal["avg", "max"] = "avg"
    dropout: float = 0.0
    feature_channels: int | None = None
    pre_kernel_size: int = 1
    groups: int = 1
    base_width: int = 64

    def __post_init__(self) -> None:
        self.stage_specs = _normalize_stage_specs(self.stage_specs)
        if self.stack_type not in {"resnet", "resnext", "darknet"}:
            raise ValueError("stack_type must be one of 'resnet', 'resnext', or 'darknet'")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.stem_channels <= 0:
            raise ValueError("stem_channels must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if not callable(self.norm):
            raise TypeError("norm must be callable")
        if not callable(self.act):
            raise TypeError("act must be callable")
        if self.split_ratio <= 0 or self.split_ratio >= 1:
            raise ValueError("split_ratio must be in the range (0, 1)")
        if self.global_pool not in {"avg", "max"}:
            raise ValueError("global_pool must be either 'avg' or 'max'")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")
        if self.feature_channels is not None and self.feature_channels <= 0:
            raise ValueError("feature_channels must be greater than 0 when provided")
        if self.pre_kernel_size <= 0:
            raise ValueError("pre_kernel_size must be greater than 0")
        if self.groups <= 0:
            raise ValueError("groups must be greater than 0")
        if self.base_width <= 0:
            raise ValueError("base_width must be greater than 0")


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
    def __init__(self, config: CSPNetConfig) -> None:
        super().__init__()
        self.config = config

        if config.stack_type == "darknet":
            block_stack_fn = _darknet_stack_factory(
                _DarknetBottleneck,
                norm=config.norm,
                act=config.act,
            )
        else:
            block_stack_fn = _resnet_stack_factory(
                _Bottleneck,
                norm=config.norm,
                act=config.act,
                groups=config.groups,
                base_width=config.base_width,
            )

        self.stem = nn.Sequential(
            _ConvBNAct(
                config.in_channels,
                config.stem_channels,
                k=3,
                s=2,
                norm=config.norm,
                act=config.act,
            ),
            _ConvBNAct(
                config.stem_channels,
                config.stem_channels,
                k=3,
                s=1,
                norm=config.norm,
                act=config.act,
            ),
        )

        stages = []
        in_ch = config.stem_channels
        for stage_width, num_layers, downsample in config.stage_specs:
            stages.append(
                _CSPStage(
                    in_ch,
                    stage_width,
                    num_layers,
                    block_stack_fn,
                    split_ratio=config.split_ratio,
                    downsample=downsample,
                    norm=config.norm,
                    act=config.act,
                    pre_kernel_size=config.pre_kernel_size,
                )
            )
            in_ch = stage_width
        self.stages = nn.Sequential(*stages)

        if config.feature_channels is not None and config.feature_channels != in_ch:
            self.pre_head = _ConvBNAct(
                in_ch,
                config.feature_channels,
                k=1,
                s=1,
                p=0,
                norm=config.norm,
                act=config.act,
            )
            in_ch = config.feature_channels
        else:
            self.pre_head = nn.Identity()

        self.num_classes = config.num_classes
        self.head_pool = (
            nn.AdaptiveAvgPool2d((1, 1))
            if config.global_pool == "avg"
            else nn.AdaptiveMaxPool2d((1, 1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity(),
            nn.Linear(in_ch, config.num_classes),
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


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_cspnet_config(
    *,
    stage_specs: tuple[tuple[int, int, bool], ...],
    stack_type: Literal["resnet", "resnext", "darknet"],
    num_classes: int,
    stem_channels: int,
    split_ratio: float,
    **kwargs,
) -> CSPNetConfig:
    return CSPNetConfig(
        stage_specs=stage_specs,
        stack_type=stack_type,
        num_classes=num_classes,
        stem_channels=stem_channels,
        split_ratio=split_ratio,
        **kwargs,
    )


@register_model
def csp_resnet_50(
    num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 64, **kwargs
) -> CSPNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"stage_specs", "stack_type", "groups", "base_width", "feature_channels"},
        "factory variants do not allow overriding preset stage_specs, stack_type, groups, base_width, or feature_channels",
    )
    config = _build_cspnet_config(
        stage_specs=(
            (256, 3, False),
            (512, 4, True),
            (1024, 6, True),
            (2048, 3, True),
        ),
        stack_type="resnet",
        num_classes=num_classes,
        stem_channels=stem_channels,
        split_ratio=split_ratio,
        groups=1,
        base_width=64,
        feature_channels=1024,
        **kwargs,
    )
    return CSPNet(config)


@register_model
def csp_resnext_50_32x4d(
    num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 64, **kwargs
) -> CSPNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"stage_specs", "stack_type", "groups", "base_width", "feature_channels"},
        "factory variants do not allow overriding preset stage_specs, stack_type, groups, base_width, or feature_channels",
    )
    config = _build_cspnet_config(
        stage_specs=(
            (256, 3, False),
            (512, 4, True),
            (1024, 6, True),
            (2048, 3, True),
        ),
        stack_type="resnext",
        num_classes=num_classes,
        stem_channels=stem_channels,
        split_ratio=split_ratio,
        groups=32,
        base_width=4,
        feature_channels=1024,
        **kwargs,
    )
    return CSPNet(config)


@register_model
def csp_darknet_53(
    num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 32, **kwargs
) -> CSPNet:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"stage_specs", "stack_type", "feature_channels", "pre_kernel_size"},
        "factory variants do not allow overriding preset stage_specs, stack_type, feature_channels, or pre_kernel_size",
    )
    config = _build_cspnet_config(
        stage_specs=(
            (64, 1, True),
            (128, 2, True),
            (256, 8, True),
            (512, 8, True),
            (1024, 4, True),
        ),
        stack_type="darknet",
        num_classes=num_classes,
        stem_channels=stem_channels,
        split_ratio=split_ratio,
        feature_channels=1024,
        pre_kernel_size=3,
        **kwargs,
    )
    return CSPNet(config)
