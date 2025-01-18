from typing import Type

import lucid.nn as nn
import math

import lucid
from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "EfficientNet",
    "EfficientNet_V2",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "efficientnet_v2_s",
    "efficientnet_v2_m",
    "efficientnet_v2_l",
    "efficientnet_v2_xl",
]


def _make_divisible(v: int, divisor: int, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)

    if new_v < 0.9 * v:
        new_v += divisor

    return int(new_v)


class _SEBlock(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excite = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.Swish(),
            nn.Linear(mid_channels, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze(x).reshape(x.shape[0], -1)
        x = self.excite(x).unsqueeze(axis=(-1, -2))
        return x


class _MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        se_scale: int = 4,
        expansion: int = 6,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.p = p if in_channels == out_channels else 1.0
        self.shortcut = stride == 1 and in_channels == out_channels
        self.expansion = expansion

        self.expand = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * expansion, eps=1e-3),
            nn.Swish(),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels * expansion,
                in_channels * expansion,
                kernel_size=kernel_size,
                padding="same",
                groups=in_channels * expansion,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * expansion, eps=1e-3),
            nn.Swish(),
        )

        self.se = _SEBlock(in_channels * expansion, max(1, int(in_channels / se_scale)))
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * expansion, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if not lucid.random.bernoulli(self.p).item():
                return x

        x_shortcut = x
        x_residual = self.expand(x) if self.expansion != 1 else x
        x_residual = self.residual(x_residual)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x
        return x


class EfficientNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_coef: float = 1.0,
        depth_coef: float = 1.0,
        scale: float = 1.0,
        dropout: float = 0.2,
        se_scale: int = 4,
        stochastic_depth: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]
        expands = [1, 6, 6, 6, 6, 6, 6]

        depth = depth_coef
        width = width_coef

        channels = [_make_divisible(ch * width, 8) for ch in channels]
        repeats = [math.ceil(rep * depth) for rep in repeats]

        if stochastic_depth:
            self.p = p
            self.step = 0.5 / (sum(repeats - 1))
        else:
            self.p = 1.0
            self.step = 0.0

        self.upsample = nn.Upsample(
            scale_factor=scale, mode="bilinear", align_corners=False
        )

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], eps=1e-3),
        )

        for i in range(7):
            block = self._make_block(
                _MBConv,
                repeats[i],
                channels[i],
                channels[i + 1],
                kernel_sizes[i],
                strides[i],
                expands[i],
                se_scale,
            )
            self.add_module(f"stage{i + 2}", block)

        self.stage9 = nn.Sequential(
            nn.Conv2d(channels[7], channels[8], kernel_size=1, bias=False),
            nn.BatchNorm2d(channels[8], eps=1e-3),
            nn.Swish(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(channels[8], num_classes)

    def _make_block(
        self,
        block: Type[nn.Module],
        repeats: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion: int,
        se_scale: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    se_scale,
                    expansion,
                    self.p,
                )
            )
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.upsample(x)
        for i in range(1, 10):
            x = getattr(self, f"stage{i}")(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


class _FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        se_scale: int = 0,
        expansion: int = 4,
        drop_prob: float = 0.2,
    ) -> None:
        super().__init__()
        self.drop_prob = drop_prob
        self.has_shortcut = stride == 1 and in_channels == out_channels

        mid_channels = in_channels * expansion
        layers = []

        if expansion != 1:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size,
                    stride,
                    padding="same",
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(mid_channels, eps=1e-3))
            layers.append(nn.Swish())

            layers.append(
                nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
            )
            layers.append(nn.BatchNorm2d(out_channels, eps=1e-3))

        else:
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding="same",
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(out_channels, eps=1e-3))
            layers.append(nn.Swish())

        self.fused_conv = nn.Sequential(*layers)
        if se_scale > 0:
            self.se = _SEBlock(out_channels, max(1, in_channels // se_scale))
        else:
            self.se = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if not lucid.random.bernoulli(self.drop_prob).item():
                return x

        out = self.fused_conv(x)
        out = self.se(out)
        if self.has_shortcut:
            out += x

        return out


class EfficientNet_V2(nn.Module):
    def __init__(
        self,
        block_cfg: list,
        num_classes: int = 1000,
        dropout: float = 0.2,
        drop_path_rate: float = 0.2,
    ) -> None:
        super().__init__()
        stem_out = block_cfg[0][1]
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_out, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out, eps=1e-3),
            nn.Swish(),
        )

        block_layers = []
        total_repeats = sum(cfg[5] for cfg in block_cfg)
        dp_index = 0

        in_channels = stem_out
        for fused, cout, k, s, e, r, se_scale in block_cfg:
            for i in range(r):
                drop_prob = 1.0 - (dp_index / total_repeats) * drop_path_rate
                dp_index += 1

                if fused:
                    block = _FusedMBConv(
                        in_channels, cout, k, s if i == 0 else 1, se_scale, e, drop_prob
                    )
                else:
                    block = _MBConv(
                        in_channels, cout, k, s if i == 0 else 1, se_scale, e, drop_prob
                    )

                block_layers.append(block)
                in_channels = cout

        self.features = nn.Sequential(*block_layers)

        head_ch = in_channels * 4
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, head_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_ch, eps=1e-3),
            nn.Swish(),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(head_ch, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.avgpool(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


@register_model
def efficientnet_b0(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=1.0,
        depth_coef=1.0,
        scale=224 / 224,
        dropout=0.2,
        **kwargs,
    )


@register_model
def efficientnet_b1(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=1.0,
        depth_coef=1.1,
        scale=240 / 224,
        dropout=0.2,
        **kwargs,
    )


@register_model
def efficientnet_b2(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=1.1,
        depth_coef=1.2,
        scale=260 / 224,
        dropout=0.3,
        **kwargs,
    )


@register_model
def efficientnet_b3(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=1.2,
        depth_coef=1.4,
        scale=300 / 224,
        dropout=0.3,
        **kwargs,
    )


@register_model
def efficientnet_b4(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=1.4,
        depth_coef=1.8,
        scale=380 / 224,
        dropout=0.4,
        **kwargs,
    )


@register_model
def efficientnet_b5(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=1.6,
        depth_coef=2.2,
        scale=456 / 224,
        dropout=0.4,
        **kwargs,
    )


@register_model
def efficientnet_b6(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=1.8,
        depth_coef=2.6,
        scale=528 / 224,
        dropout=0.5,
        **kwargs,
    )


@register_model
def efficientnet_b7(num_classes: int = 1000, **kwargs) -> EfficientNet:
    return EfficientNet(
        num_classes,
        width_coef=2.0,
        depth_coef=3.1,
        scale=600 / 224,
        dropout=0.5,
        **kwargs,
    )


@register_model
def efficientnet_v2_s(num_classes: int = 1000, **kwargs) -> EfficientNet_V2:
    cfg = [
        [True, 24, 3, 2, 1, 2, 0],
        [True, 48, 3, 2, 4, 4, 0],
        [True, 64, 3, 2, 4, 4, 0],
        [False, 128, 3, 2, 4, 6, 4],
        [False, 160, 3, 1, 6, 9, 4],
        [False, 256, 3, 2, 6, 15, 4],
    ]
    return EfficientNet_V2(cfg, num_classes, dropout=0.2, drop_path_rate=0.2, **kwargs)


@register_model
def efficientnet_v2_m(num_classes: int = 1000, **kwargs) -> EfficientNet_V2:
    cfg = [
        [True, 24, 3, 2, 1, 3, 0],
        [True, 48, 3, 2, 4, 5, 0],
        [True, 80, 3, 2, 4, 5, 0],
        [False, 160, 3, 2, 4, 7, 4],
        [False, 176, 3, 1, 6, 14, 4],
        [False, 304, 3, 2, 6, 18, 4],
        [False, 512, 3, 1, 6, 5, 4],
    ]
    return EfficientNet_V2(cfg, num_classes, dropout=0.3, drop_path_rate=0.2, **kwargs)


@register_model
def efficientnet_v2_l(num_classes: int = 1000, **kwargs) -> EfficientNet_V2:
    cfg = [
        [True, 32, 3, 2, 1, 4, 0],
        [True, 64, 3, 2, 4, 7, 0],
        [True, 96, 3, 2, 4, 7, 0],
        [False, 192, 3, 2, 4, 10, 4],
        [False, 224, 3, 1, 6, 19, 4],
        [False, 384, 3, 2, 6, 25, 4],
        [False, 640, 3, 1, 6, 7, 4],
    ]
    return EfficientNet_V2(cfg, num_classes, dropout=0.4, drop_path_rate=0.3, **kwargs)


@register_model
def efficientnet_v2_xl(num_classes: int = 1000, **kwargs) -> EfficientNet_V2:
    cfg = [
        [True, 32, 3, 2, 1, 4, 0],
        [True, 64, 3, 2, 4, 8, 0],
        [True, 96, 3, 2, 4, 8, 0],
        [False, 192, 3, 2, 4, 16, 4],
        [False, 256, 3, 1, 6, 24, 4],
        [False, 512, 3, 2, 6, 32, 4],
        [False, 640, 3, 1, 6, 8, 4],
    ]
    return EfficientNet_V2(cfg, num_classes, dropout=0.5, drop_path_rate=0.4, **kwargs)
