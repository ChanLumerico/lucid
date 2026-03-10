from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor
from lucid.types import _Scalar
from lucid.models.base import PreTrainedModelMixin

__all__ = [
    "MobileNet",
    "MobileNetConfig",
    "MobileNet_V2",
    "MobileNetV2Config",
    "MobileNet_V3",
    "MobileNetV3Config",
    "MobileNet_V4",
    "MobileNetV4Config",
    "mobilenet",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
    "mobilenet_v4_conv_small",
    "mobilenet_v4_conv_medium",
    "mobilenet_v4_conv_large",
    "mobilenet_v4_hybrid_medium",
    "mobilenet_v4_hybrid_large",
]


_MOBILENET_V2_STAGE_CONFIGS = (
    (32, 16, 1, 1, 1),
    (16, 24, 6, 2, 2),
    (24, 32, 6, 3, 2),
    (32, 64, 6, 4, 2),
    (64, 96, 6, 3, 1),
    (96, 160, 6, 3, 2),
    (160, 320, 6, 1, 1),
)

_MOBILENET_V3_SMALL_CFG = (
    (3, 16, 16, True, False, 2, 2),
    (3, 72, 24, False, False, 2, 4),
    (3, 88, 24, False, False, 1, 4),
    (5, 96, 40, True, True, 2, 4),
    (5, 240, 40, True, True, 1, 4),
    (5, 240, 40, True, True, 1, 4),
    (5, 120, 48, True, True, 1, 4),
    (5, 144, 48, True, True, 1, 4),
    (5, 288, 96, True, True, 2, 4),
    (5, 576, 96, True, True, 1, 4),
    (5, 576, 96, True, True, 1, 4),
)

_MOBILENET_V3_LARGE_CFG = (
    (3, 16, 16, False, False, 1, 4),
    (3, 64, 24, False, False, 2, 4),
    (3, 72, 24, False, False, 1, 4),
    (5, 72, 40, True, False, 2, 4),
    (5, 120, 40, True, False, 1, 4),
    (5, 120, 40, True, False, 1, 4),
    (3, 240, 80, False, True, 2, 4),
    (3, 200, 80, False, True, 1, 4),
    (3, 184, 80, False, True, 1, 4),
    (3, 184, 80, False, True, 1, 4),
    (3, 480, 112, True, True, 1, 4),
    (3, 672, 112, True, True, 1, 4),
    (5, 672, 160, True, True, 2, 4),
    (5, 960, 160, True, True, 1, 4),
    (5, 960, 160, True, True, 1, 4),
)


@dataclass
class MobileNetConfig:
    width_multiplier: float = 1.0
    num_classes: int = 1000
    in_channels: int = 3

    def __post_init__(self) -> None:
        if self.width_multiplier <= 0:
            raise ValueError("width_multiplier must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")


@dataclass
class MobileNetV2Config:
    stage_configs: tuple[tuple[int, int, int, int, int], ...] | list[tuple[int, int, int, int, int]]
    num_classes: int = 1000
    in_channels: int = 3
    stem_channels: int = 32
    last_channels: int = 1280
    dropout: float = 0.2

    def __post_init__(self) -> None:
        self.stage_configs = tuple(tuple(stage) for stage in self.stage_configs)
        if len(self.stage_configs) == 0:
            raise ValueError("stage_configs must contain at least one stage")
        for stage in self.stage_configs:
            if len(stage) != 5:
                raise ValueError(
                    "each stage config must contain exactly 5 values: (in_channels, out_channels, expansion, repeats, stride)"
                )
            if any(not isinstance(value, int) or value <= 0 for value in stage):
                raise ValueError("stage config values must be positive integers")

        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.stem_channels <= 0:
            raise ValueError("stem_channels must be greater than 0")
        if self.last_channels <= 0:
            raise ValueError("last_channels must be greater than 0")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")


@dataclass
class MobileNetV3Config:
    bottleneck_cfg: tuple[tuple[object, ...], ...] | list[list[object]]
    last_channels: int
    num_classes: int = 1000
    in_channels: int = 3
    stem_channels: int = 16
    dropout: float = 0.2

    def __post_init__(self) -> None:
        self.bottleneck_cfg = tuple(tuple(spec) for spec in self.bottleneck_cfg)
        if len(self.bottleneck_cfg) == 0:
            raise ValueError("bottleneck_cfg must contain at least one bottleneck spec")
        for spec in self.bottleneck_cfg:
            if len(spec) != 7:
                raise ValueError(
                    "each bottleneck spec must contain exactly 7 values"
                )
            kernel_size, mid_channels, out_channels, use_se, use_hswish, stride, se_reduction = spec
            int_values = (kernel_size, mid_channels, out_channels, stride, se_reduction)
            if any(not isinstance(value, int) or value <= 0 for value in int_values):
                raise ValueError(
                    "bottleneck spec kernel, channel, stride, and reduction values must be positive integers"
                )
            if not isinstance(use_se, bool) or not isinstance(use_hswish, bool):
                raise TypeError(
                    "bottleneck spec use_se and use_hswish values must be booleans"
                )

        if self.last_channels <= 0:
            raise ValueError("last_channels must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.stem_channels <= 0:
            raise ValueError("stem_channels must be greater than 0")
        if self.dropout < 0 or self.dropout >= 1:
            raise ValueError("dropout must be in the range [0, 1)")


@dataclass
class MobileNetV4Config:
    cfg: dict[str, dict[str, Any]]
    num_classes: int = 1000

    def __post_init__(self) -> None:
        if not isinstance(self.cfg, dict):
            raise TypeError("cfg must be a dictionary")
        self.cfg = deepcopy(self.cfg)

        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")

        required_keys = ("conv0", "layer1", "layer2", "layer3", "layer4", "layer5")
        missing_keys = [key for key in required_keys if key not in self.cfg]
        if missing_keys:
            raise ValueError(
                f"cfg must define conv0 and layer1-layer5 blocks; missing {missing_keys}"
            )

        valid_block_names = {"convbn", "uib", "fused_ib"}
        for key in required_keys:
            layer_spec = self.cfg[key]
            if not isinstance(layer_spec, dict):
                raise TypeError(f"{key} spec must be a dictionary")
            if "block_name" not in layer_spec or "num_blocks" not in layer_spec or "block_specs" not in layer_spec:
                raise ValueError(
                    f"{key} spec must contain block_name, num_blocks, and block_specs"
                )
            block_name = layer_spec["block_name"]
            num_blocks = layer_spec["num_blocks"]
            block_specs = layer_spec["block_specs"]
            if block_name not in valid_block_names:
                raise ValueError(
                    f"{key} block_name must be one of {sorted(valid_block_names)}"
                )
            if not isinstance(num_blocks, int) or num_blocks <= 0:
                raise ValueError(f"{key} num_blocks must be a positive integer")
            if not isinstance(block_specs, list) or len(block_specs) != num_blocks:
                raise ValueError(
                    f"{key} block_specs must be a list whose length matches num_blocks"
                )


class _Depthwise(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(x))


class MobileNet(nn.Module, PreTrainedModelMixin):
    def __init__(self, config: MobileNetConfig) -> None:
        super().__init__()
        self.config = config
        alpha = config.width_multiplier

        self.conv1 = nn.ConvBNReLU2d(
            config.in_channels, int(32 * alpha), kernel_size=3, stride=2, padding=1
        )
        self.conv2 = _Depthwise(int(32 * alpha), int(64 * alpha))

        self.conv3 = nn.Sequential(
            _Depthwise(int(64 * alpha), int(128 * alpha), stride=2),
            _Depthwise(int(128 * alpha), int(128 * alpha), stride=1),
        )
        self.conv4 = nn.Sequential(
            _Depthwise(int(128 * alpha), int(256 * alpha), stride=2),
            _Depthwise(int(256 * alpha), int(256 * alpha), stride=1),
        )

        self.conv5 = nn.Sequential(
            _Depthwise(int(256 * alpha), int(512 * alpha), stride=2),
            *[
                _Depthwise(int(512 * alpha), int(512 * alpha), stride=1)
                for _ in range(5)
            ],
        )
        self.conv6 = _Depthwise(int(512 * alpha), int(1024 * alpha), stride=2)
        self.conv7 = _Depthwise(int(1024 * alpha), int(1024 * alpha), stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(int(1024 * alpha), config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class _InvertedBottleneck(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, t: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        expand = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(),
        )
        depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels * t,
                in_channels * t,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels * t,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(),
        )
        pointwise = nn.Sequential(
            nn.Conv2d(in_channels * t, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        residual_list = []
        if t > 1:
            residual_list += [expand]
        residual_list += [depthwise, pointwise]

        self.residual = nn.Sequential(*residual_list)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride == 1 and self.in_channels == self.out_channels:
            out = self.residual(x) + x
        else:
            out = self.residual(x)
        return out


class MobileNet_V2(nn.Module, PreTrainedModelMixin):
    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__()
        self.config = config

        self.conv_first = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.stem_channels),
            nn.ReLU6(),
        )

        self.bottlenecks = nn.Sequential(
            *[
                self._make_stage(
                    in_channels,
                    out_channels,
                    t=expand_ratio,
                    n=repeats,
                    stride=stride,
                )
                for in_channels, out_channels, expand_ratio, repeats, stride in config.stage_configs
            ]
        )

        self.conv_last = nn.Sequential(
            nn.Conv2d(
                config.stage_configs[-1][1],
                config.last_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.last_channels),
            nn.ReLU6(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(config.last_channels, config.num_classes),
        )

    def _make_stage(
        self, in_channels: int, out_channels: int, t: int, n: int, stride: int = 1
    ) -> nn.Sequential:
        layers = [_InvertedBottleneck(in_channels, out_channels, t, stride=stride)]
        in_channels = out_channels
        for _ in range(n - 1):
            layers.append(_InvertedBottleneck(in_channels, out_channels, t, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_first(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


class _SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)

        self.relu = nn.ReLU()
        self.hsigmoid = nn.HardSigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.avgpool(x).squeeze(axis=(-1, -2))
        y = self.relu(self.fc1(y))
        y = self.hsigmoid(self.fc2(y))

        y = y.unsqueeze(axis=(-1, -2))
        out = x * y
        return out


class _InvertedBottleneck_V3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hswish: bool,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        self.do_skip = stride == 1 and in_channels == out_channels

        expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels, momentum=0.99),
            nn.HardSwish() if use_hswish else nn.ReLU(),
        )

        depthwise = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size,
                stride,
                padding="same",
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels, momentum=0.99),
            nn.HardSwish() if use_hswish else nn.ReLU(),
        )

        se_block = _SEBlock(mid_channels, se_reduction) if use_se else None

        pointwise = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=0.99),
        )

        layers = []
        if in_channels < mid_channels:
            layers.append(expand)
        layers.append(depthwise)

        if se_block is not None:
            layers.append(se_block)
        layers.append(pointwise)

        self.residual = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.do_skip:
            out = self.residual(x) + x
        else:
            out = self.residual(x)
        return out


class MobileNet_V3(nn.Module, PreTrainedModelMixin):
    def __init__(self, config: MobileNetV3Config) -> None:
        super().__init__()
        self.config = config

        self.conv_first = nn.Sequential(
            nn.Conv2d(
                config.in_channels,
                config.stem_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(config.stem_channels),
            nn.HardSwish(),
        )

        in_channels = config.stem_channels
        bottleneck_layers = []
        for (
            kernel_size,
            mid_channels,
            out_channels,
            use_se,
            use_hswish,
            stride,
            se_reduction,
        ) in config.bottleneck_cfg:
            bottleneck_layers.append(
                _InvertedBottleneck_V3(
                    in_channels,
                    mid_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    use_se,
                    use_hswish,
                    se_reduction,
                )
            )
            in_channels = out_channels
        self.bottlenecks = nn.Sequential(*bottleneck_layers)
        last_mid_channels = config.bottleneck_cfg[-1][1]

        self.conv_last = nn.Sequential(
            nn.Conv2d(in_channels, last_mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_mid_channels),
            nn.HardSwish(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Sequential(
            nn.Linear(last_mid_channels, config.last_channels),
            nn.HardSwish(),
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(config.last_channels, config.num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_first(x)
        x = self.bottlenecks(x)
        x = self.conv_last(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x


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


class _UniversalInvertedBottleneck(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        start_dw_kernel_size: int,
        mid_dw_kernel_size: int,
        mid_dw_downsample: bool,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        if self.start_dw_kernel_size:
            stride_ = stride if not mid_dw_downsample else 1
            self.start_dw = _make_conv_block(
                in_channels,
                in_channels,
                kernel_size=start_dw_kernel_size,
                stride=stride_,
                groups=in_channels,
                act=False,
            )

        expand_filters = _make_divisible(in_channels * expand_ratio, divisor=8)
        self.expand_conv = _make_conv_block(in_channels, expand_filters, kernel_size=1)

        self.mid_dw_kernel_size = mid_dw_kernel_size
        if self.mid_dw_kernel_size:
            stride_ = stride if mid_dw_downsample else 1
            self.middle_dw = _make_conv_block(
                expand_filters,
                expand_filters,
                kernel_size=mid_dw_kernel_size,
                stride=stride_,
                groups=expand_filters,
            )

        self.proj_conv = _make_conv_block(
            expand_filters, out_channels, kernel_size=1, act=False
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.start_dw_kernel_size:
            x = self.start_dw(x)

        x = self.expand_conv(x)
        if self.mid_dw_kernel_size:
            x = self.middle_dw(x)

        x = self.proj_conv(x)
        return x


class _MultiQueryAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_stride: int,
        query_w_stride: int,
        kv_stride: int,
        dw_kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_stride = query_h_stride
        self.query_w_stride = query_w_stride
        self.kv_stride = kv_stride
        self.dw_kernel_size = dw_kernel_size
        self.dropout = dropout

        self.head_dim = key_dim // num_heads

        if self.query_h_stride > 1 or query_w_stride > 1:
            self.query_downsample_norm = nn.BatchNorm2d(in_channels)
        self.query_proj = _make_conv_block(
            in_channels, num_heads * key_dim, kernel_size=1, norm=False, act=False
        )

        if self.kv_stride > 1:
            self.key_dw_conv = _make_conv_block(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                groups=in_channels,
                act=False,
            )
            self.value_dw_conv = _make_conv_block(
                in_channels,
                in_channels,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                groups=in_channels,
                act=False,
            )

        self.key_proj = _make_conv_block(
            in_channels, key_dim, kernel_size=1, norm=False, act=False
        )
        self.value_proj = _make_conv_block(
            in_channels, key_dim, kernel_size=1, norm=False, act=False
        )
        self.output_proj = _make_conv_block(
            num_heads * key_dim, in_channels, kernel_size=1, norm=False, act=False
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        if self.query_h_stride > 1 or self.query_w_stride > 1:
            q = F.avg_pool2d(
                x,
                kernel_size=(self.query_h_stride, self.query_w_stride),
                stride=(self.query_h_stride, self.query_w_stride),
            )
            q = self.query_downsample_norm(q)
            q = self.query_proj(q)
        else:
            q = self.query_proj(x)

        px = q.shape[2]
        q = q.reshape(N, self.num_heads, -1, self.key_dim)

        if self.kv_stride > 1:
            k = self.key_dw_conv(x)
            k = self.key_proj(k)
            v = self.value_dw_conv(x)
            v = self.value_proj(v)
        else:
            k = self.key_proj(x)
            v = self.value_proj(x)

        k = k.reshape(N, 1, self.key_dim, -1)
        v = v.reshape(N, 1, -1, self.key_dim)

        attn_score = (q @ k) / (self.head_dim**0.5)
        attn_score = self.dropout(attn_score)
        attn_score = F.softmax(attn_score, axis=-1)

        context = attn_score @ v
        context = context.reshape(N, self.num_heads * self.key_dim, px, px)

        out = self.output_proj(context)
        return out


class _LayerScale(nn.Module):
    def __init__(self, in_channels: int, init_value: _Scalar) -> None:
        super().__init__()
        self.init_value = init_value
        self.gamma = nn.Parameter(self.init_value * lucid.ones(in_channels, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_heads: int,
        key_dim: int,
        value_dim: int,
        query_h_stride: int,
        query_w_stride: int,
        kv_stride: int,
        use_layer_scale: bool,
        use_multi_query: bool,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.query_h_stride = query_h_stride
        self.query_w_stride = query_w_stride
        self.kv_stride = kv_stride
        self.use_layer_scale = use_layer_scale
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual

        self.input_norm = nn.BatchNorm2d(in_channels)
        if self.use_multi_query:
            self.multi_query_attention = _MultiQueryAttention(
                in_channels,
                num_heads,
                key_dim,
                value_dim,
                query_h_stride,
                query_w_stride,
                kv_stride,
            )
        else:
            self.multi_head_attention = nn.MultiHeadAttention(
                in_channels, num_heads, kdim=key_dim
            )

        if self.use_layer_scale:
            self.layer_scale_init_value = 1e-5
            self.layer_scale = _LayerScale(in_channels, self.layer_scale_init_value)

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.input_norm(x)

        if self.use_multi_query:
            x = self.multi_query_attention(x)
        else:
            x = self.multi_head_attention(x, x, x)

        if self.use_layer_scale:
            x = self.layer_scale(x)
        if self.use_residual:
            x += shortcut

        return x


def build_v4_blocks(layer_spec: dict) -> nn.Sequential:
    if not layer_spec.get("block_name"):
        return nn.Sequential()

    block_names = layer_spec["block_name"]
    layers = nn.Sequential()

    if block_names == "convbn":
        schema = ["in_channels", "out_channels", "kernel_size", "stride"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema, layer_spec["block_specs"][i]))
            layers.add_module(f"convbn_{i}", _make_conv_block(**args))

    elif block_names == "uib":
        schema = [
            "in_channels",
            "out_channels",
            "start_dw_kernel_size",
            "mid_dw_kernel_size",
            "mid_dw_downsample",
            "stride",
            "expand_ratio",
            "mha",
        ]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema, layer_spec["block_specs"][i]))
            mha = args.pop("mha") if "mha" in args else 0

            layers.add_module(f"uib_{i}", _UniversalInvertedBottleneck(**args))
            if mha:
                mha_schema = [
                    "in_channels",
                    "num_heads",
                    "key_dim",
                    "value_dim",
                    "query_h_stride",
                    "query_w_stride",
                    "kv_stride",
                    "use_layer_scale",
                    "use_multi_query",
                    "use_residual",
                ]
                args = dict(zip(mha_schema, [args["out_channels"]] + mha))
                layers.add_module(f"mha_{i}", _MultiHeadSelfAttention(**args))

    elif block_names == "fused_ib":
        schema = ["in_channels", "out_channels", "stride", "expand_ratio", "act"]
        for i in range(layer_spec["num_blocks"]):
            args = dict(zip(schema, layer_spec["block_specs"][i]))
            layers.add_module(f"fused_ib_{i}", _InvertedResidual(**args))

    else:
        raise NotImplementedError

    return layers


class MobileNet_V4(nn.Module):
    def __init__(self, config: MobileNetV4Config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.conv0 = build_v4_blocks(config.cfg["conv0"])

        self.layers = nn.ModuleList()
        for i in range(1, 6):
            self.layers.append(build_v4_blocks(config.cfg[f"layer{i}"]))

        last_channels = config.cfg["layer5"]["block_specs"][-1][1]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(last_channels, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv0(x)
        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


def _build_mobilenet_config(
    *,
    width_multiplier: float,
    num_classes: int,
    kwargs: dict[str, object] | None = None,
) -> MobileNetConfig:
    kwargs = {} if kwargs is None else dict(kwargs)
    if "width_multiplier" in kwargs:
        raise TypeError("mobilenet does not allow width_multiplier inside kwargs")

    return MobileNetConfig(
        width_multiplier=width_multiplier,
        num_classes=num_classes,
        **kwargs,
    )


def _build_mobilenet_v2_config(
    *,
    num_classes: int,
    kwargs: dict[str, object] | None = None,
) -> MobileNetV2Config:
    kwargs = {} if kwargs is None else dict(kwargs)
    locked_fields = {"stage_configs", "stem_channels", "last_channels"}
    if locked_fields & kwargs.keys():
        raise TypeError(
            "factory variants do not allow overriding preset stage_configs, stem_channels, or last_channels"
        )

    return MobileNetV2Config(
        stage_configs=_MOBILENET_V2_STAGE_CONFIGS,
        num_classes=num_classes,
        **kwargs,
    )


def _build_mobilenet_v3_config(
    *,
    bottleneck_cfg: tuple[tuple[object, ...], ...],
    last_channels: int,
    num_classes: int,
    kwargs: dict[str, object] | None = None,
) -> MobileNetV3Config:
    kwargs = {} if kwargs is None else dict(kwargs)
    locked_fields = {"bottleneck_cfg", "last_channels", "stem_channels"}
    if locked_fields & kwargs.keys():
        raise TypeError(
            "factory variants do not allow overriding preset bottleneck_cfg, last_channels, or stem_channels"
        )

    return MobileNetV3Config(
        bottleneck_cfg=bottleneck_cfg,
        last_channels=last_channels,
        num_classes=num_classes,
        **kwargs,
    )


def _build_mobilenet_v4_config(
    *,
    cfg: dict[str, dict[str, Any]],
    num_classes: int,
    kwargs: dict[str, object] | None = None,
) -> MobileNetV4Config:
    kwargs = {} if kwargs is None else dict(kwargs)
    if "cfg" in kwargs:
        raise TypeError("factory variants do not allow overriding preset cfg")

    return MobileNetV4Config(cfg=cfg, num_classes=num_classes, **kwargs)


@register_model
def mobilenet(
    width_multiplier: float = 1.0, num_classes: int = 1000, **kwargs
) -> MobileNet:
    config = _build_mobilenet_config(
        width_multiplier=width_multiplier,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return MobileNet(config)


@register_model
def mobilenet_v2(num_classes: int = 1000, **kwargs) -> MobileNet_V2:
    config = _build_mobilenet_v2_config(num_classes=num_classes, kwargs=kwargs)
    return MobileNet_V2(config)


@register_model
def mobilenet_v3_small(num_classes: int = 1000, **kwargs) -> MobileNet_V3:
    config = _build_mobilenet_v3_config(
        bottleneck_cfg=_MOBILENET_V3_SMALL_CFG,
        last_channels=1024,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return MobileNet_V3(config)


@register_model
def mobilenet_v3_large(num_classes: int = 1000, **kwargs) -> MobileNet_V3:
    config = _build_mobilenet_v3_config(
        bottleneck_cfg=_MOBILENET_V3_LARGE_CFG,
        last_channels=1280,
        num_classes=num_classes,
        kwargs=kwargs,
    )
    return MobileNet_V3(config)


@register_model
def mobilenet_v4_conv_small(num_classes: int = 1000, **kwargs) -> MobileNet_V4:
    cfg = dict(
        conv0={
            "block_name": "convbn",
            "num_blocks": 1,
            "block_specs": [[3, 32, 3, 2]],
        },
        layer1={
            "block_name": "convbn",
            "num_blocks": 2,
            "block_specs": [[32, 32, 3, 2], [32, 32, 1, 1]],
        },
        layer2={
            "block_name": "convbn",
            "num_blocks": 2,
            "block_specs": [[32, 96, 3, 2], [96, 64, 1, 1]],
        },
        layer3={
            "block_name": "uib",
            "num_blocks": 6,
            "block_specs": [
                [64, 96, 5, 5, True, 2, 3],
                [96, 96, 0, 3, True, 1, 2],
                [96, 96, 0, 3, True, 1, 2],
                [96, 96, 0, 3, True, 1, 2],
                [96, 96, 0, 3, True, 1, 2],
                [96, 96, 3, 0, True, 1, 4],
            ],
        },
        layer4={
            "block_name": "uib",
            "num_blocks": 6,
            "block_specs": [
                [96, 128, 3, 3, True, 2, 6],
                [128, 128, 5, 5, True, 1, 4],
                [128, 128, 0, 5, True, 1, 4],
                [128, 128, 0, 5, True, 1, 3],
                [128, 128, 0, 3, True, 1, 4],
                [128, 128, 0, 3, True, 1, 4],
            ],
        },
        layer5={
            "block_name": "convbn",
            "num_blocks": 2,
            "block_specs": [[128, 960, 1, 1], [960, 1280, 1, 1]],
        },
    )
    config = _build_mobilenet_v4_config(cfg=cfg, num_classes=num_classes, kwargs=kwargs)
    return MobileNet_V4(config)


@register_model
def mobilenet_v4_conv_medium(num_classes: int = 1000, **kwargs) -> MobileNet_V4:
    cfg = dict(
        conv0={
            "block_name": "convbn",
            "num_blocks": 1,
            "block_specs": [[3, 32, 3, 2]],
        },
        layer1={
            "block_name": "fused_ib",
            "num_blocks": 1,
            "block_specs": [[32, 48, 2, 4.0, True]],
        },
        layer2={
            "block_name": "uib",
            "num_blocks": 2,
            "block_specs": [[48, 80, 3, 5, True, 2, 4], [80, 80, 3, 3, True, 1, 2]],
        },
        layer3={
            "block_name": "uib",
            "num_blocks": 8,
            "block_specs": [
                [80, 160, 3, 5, True, 2, 6],
                [160, 160, 3, 3, True, 1, 4],
                [160, 160, 3, 3, True, 1, 4],
                [160, 160, 3, 5, True, 1, 4],
                [160, 160, 3, 3, True, 1, 4],
                [160, 160, 3, 0, True, 1, 4],
                [160, 160, 0, 0, True, 1, 2],
                [160, 160, 3, 0, True, 1, 4],
            ],
        },
        layer4={
            "block_name": "uib",
            "num_blocks": 11,
            "block_specs": [
                [160, 256, 5, 5, True, 2, 6],
                [256, 256, 5, 5, True, 1, 4],
                [256, 256, 3, 5, True, 1, 4],
                [256, 256, 3, 5, True, 1, 4],
                [256, 256, 0, 0, True, 1, 4],
                [256, 256, 3, 0, True, 1, 4],
                [256, 256, 3, 5, True, 1, 2],
                [256, 256, 5, 5, True, 1, 4],
                [256, 256, 0, 0, True, 1, 4],
                [256, 256, 0, 0, True, 1, 4],
                [256, 256, 5, 0, True, 1, 2],
            ],
        },
        layer5={
            "block_name": "convbn",
            "num_blocks": 2,
            "block_specs": [[256, 960, 1, 1], [960, 1280, 1, 1]],
        },
    )
    config = _build_mobilenet_v4_config(cfg=cfg, num_classes=num_classes, kwargs=kwargs)
    return MobileNet_V4(config)


@register_model
def mobilenet_v4_conv_large(num_classes: int = 1000, **kwargs) -> MobileNet_V4:
    cfg = dict(
        conv0={
            "block_name": "convbn",
            "num_blocks": 1,
            "block_specs": [[3, 24, 3, 2]],
        },
        layer1={
            "block_name": "fused_ib",
            "num_blocks": 1,
            "block_specs": [[24, 48, 2, 4.0, True]],
        },
        layer2={
            "block_name": "uib",
            "num_blocks": 2,
            "block_specs": [[48, 96, 3, 5, True, 2, 4], [96, 96, 3, 3, True, 1, 4]],
        },
        layer3={
            "block_name": "uib",
            "num_blocks": 11,
            "block_specs": [
                [96, 192, 3, 5, True, 2, 4],
                [192, 192, 3, 3, True, 1, 4],
                [192, 192, 3, 3, True, 1, 4],
                [192, 192, 3, 3, True, 1, 4],
                [192, 192, 3, 5, True, 1, 4],
                [192, 192, 5, 3, True, 1, 4],
                [192, 192, 5, 3, True, 1, 4],
                [192, 192, 5, 3, True, 1, 4],
                [192, 192, 5, 3, True, 1, 4],
                [192, 192, 5, 3, True, 1, 4],
                [192, 192, 3, 0, True, 1, 4],
            ],
        },
        layer4={
            "block_name": "uib",
            "num_blocks": 13,
            "block_specs": [
                [192, 512, 5, 5, True, 2, 4],
                [512, 512, 5, 5, True, 1, 4],
                [512, 512, 5, 5, True, 1, 4],
                [512, 512, 5, 5, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 3, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 3, True, 1, 4],
                [512, 512, 5, 5, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
            ],
        },
        layer5={
            "block_name": "convbn",
            "num_blocks": 2,
            "block_specs": [[512, 960, 1, 1], [960, 1280, 1, 1]],
        },
    )
    config = _build_mobilenet_v4_config(cfg=cfg, num_classes=num_classes, kwargs=kwargs)
    return MobileNet_V4(config)


def _mha_specs(num_heads: int, key_dim: int, value_dim: int, px: int) -> list[int]:
    if px == 24:
        kv_stride = 2
    elif px == 12:
        kv_stride = 1
    query_h_stride = query_w_stride = 1
    use_layer_scale = use_multi_query = use_residual = True
    return [
        num_heads,
        key_dim,
        value_dim,
        query_h_stride,
        query_w_stride,
        kv_stride,
        use_layer_scale,
        use_multi_query,
        use_residual,
    ]


@register_model
def mobilenet_v4_hybrid_medium(num_classes: int = 1000, **kwargs) -> MobileNet_V4:
    cfg = dict(
        conv0={
            "block_name": "convbn",
            "num_blocks": 1,
            "block_specs": [[3, 32, 3, 2]],
        },
        layer1={
            "block_name": "fused_ib",
            "num_blocks": 1,
            "block_specs": [[32, 48, 2, 4.0, True]],
        },
        layer2={
            "block_name": "uib",
            "num_blocks": 2,
            "block_specs": [[48, 80, 3, 5, True, 2, 4], [80, 80, 3, 3, True, 1, 2]],
        },
        layer3={
            "block_name": "uib",
            "num_blocks": 8,
            "block_specs": [
                [80, 160, 3, 5, True, 2, 6],
                [160, 160, 0, 0, True, 1, 2],
                [160, 160, 3, 3, True, 1, 4],
                [160, 160, 3, 5, True, 1, 4, _mha_specs(4, 64, 64, 24)],
                [160, 160, 3, 3, True, 1, 4, _mha_specs(4, 64, 64, 24)],
                [160, 160, 3, 0, True, 1, 4, _mha_specs(4, 64, 64, 24)],
                [160, 160, 3, 3, True, 1, 4, _mha_specs(4, 64, 64, 24)],
                [160, 160, 3, 0, True, 1, 4],
            ],
        },
        layer4={
            "block_name": "uib",
            "num_blocks": 12,
            "block_specs": [
                [160, 256, 5, 5, True, 2, 6],
                [256, 256, 5, 5, True, 1, 4],
                [256, 256, 3, 5, True, 1, 4],
                [256, 256, 3, 5, True, 1, 4],
                [256, 256, 0, 0, True, 1, 2],
                [256, 256, 3, 5, True, 1, 2],
                [256, 256, 0, 0, True, 1, 2],
                [256, 256, 0, 0, True, 1, 4, _mha_specs(4, 64, 64, 12)],
                [256, 256, 3, 0, True, 1, 4, _mha_specs(4, 64, 64, 12)],
                [256, 256, 5, 5, True, 1, 4, _mha_specs(4, 64, 64, 12)],
                [256, 256, 5, 0, True, 1, 4, _mha_specs(4, 64, 64, 12)],
                [256, 256, 5, 0, True, 1, 4],
            ],
        },
        layer5={
            "block_name": "convbn",
            "num_blocks": 2,
            "block_specs": [[256, 960, 1, 1], [960, 1280, 1, 1]],
        },
    )
    config = _build_mobilenet_v4_config(cfg=cfg, num_classes=num_classes, kwargs=kwargs)
    return MobileNet_V4(config)


@register_model
def mobilenet_v4_hybrid_large(num_classes: int = 1000, **kwargs) -> MobileNet_V4:
    cfg = dict(
        conv0={"block_name": "convbn", "num_blocks": 1, "block_specs": [[3, 24, 3, 2]]},
        layer1={
            "block_name": "fused_ib",
            "num_blocks": 1,
            "block_specs": [[24, 48, 2, 4.0, True]],
        },
        layer2={
            "block_name": "uib",
            "num_blocks": 2,
            "block_specs": [[48, 96, 3, 5, True, 2, 4], [96, 96, 3, 3, True, 1, 4]],
        },
        layer3={
            "block_name": "uib",
            "num_blocks": 11,
            "block_specs": [
                [96, 192, 3, 5, True, 2, 4],
                [192, 192, 3, 3, True, 1, 4],
                [192, 192, 3, 3, True, 1, 4],
                [192, 192, 3, 3, True, 1, 4],
                [192, 192, 3, 5, True, 1, 4],
                [192, 192, 5, 3, True, 1, 4],
                [192, 192, 5, 3, True, 1, 4, _mha_specs(8, 48, 48, 24)],
                [192, 192, 5, 3, True, 1, 4, _mha_specs(8, 48, 48, 24)],
                [192, 192, 5, 3, True, 1, 4, _mha_specs(8, 48, 48, 24)],
                [192, 192, 5, 3, True, 1, 4, _mha_specs(8, 48, 48, 24)],
                [192, 192, 3, 0, True, 1, 4],
            ],
        },
        layer4={
            "block_name": "uib",
            "num_blocks": 14,
            "block_specs": [
                [192, 512, 5, 5, True, 2, 4],
                [512, 512, 5, 5, True, 1, 4],
                [512, 512, 5, 5, True, 1, 4],
                [512, 512, 5, 5, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 3, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 0, True, 1, 4],
                [512, 512, 5, 3, True, 1, 4],
                [512, 512, 5, 5, True, 1, 4, _mha_specs(8, 64, 64, 12)],
                [512, 512, 5, 0, True, 1, 4, _mha_specs(8, 64, 64, 12)],
                [512, 512, 5, 0, True, 1, 4, _mha_specs(8, 64, 64, 12)],
                [512, 512, 5, 0, True, 1, 4, _mha_specs(8, 64, 64, 12)],
                [512, 512, 5, 0, True, 1, 4],
            ],
        },
        layer5={
            "block_name": "convbn",
            "num_blocks": 2,
            "block_specs": [[512, 960, 1, 1], [960, 1280, 1, 1]],
        },
    )
    config = _build_mobilenet_v4_config(cfg=cfg, num_classes=num_classes, kwargs=kwargs)
    return MobileNet_V4(config)
