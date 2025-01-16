from typing import ClassVar, Type

import lucid.nn as nn

import lucid
from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "EfficientNet",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
]


class _SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.excite = nn.Sequential(
            nn.Linear(in_channels, in_channels * reduction),
            nn.Swish(),
            nn.Linear(in_channels * reduction, in_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.squeeze(x).reshape(x.shape[0], -1)
        x = self.excite(x).unsqueeze(axis=(-1, -2))
        return x


class _MBConv(nn.Module):
    expansion: ClassVar[int] = 6

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        se_scale: int = 4,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.p = p if in_channels == out_channels else 1.0
        self.shortcut = stride == 1 and in_channels == out_channels

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels * self.expansion,
                kernel_size=1,
                stride=stride,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * self.expansion, momentum=0.99, eps=1e-3),
            nn.Swish(),
            nn.Conv2d(
                in_channels * self.expansion,
                in_channels * self.expansion,
                kernel_size=kernel_size,
                padding="same",
                groups=in_channels * self.expansion,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * self.expansion, momentum=0.99, eps=1e-3),
            nn.Swish(),
        )

        self.se = _SEBlock(in_channels * self.expansion, reduction=se_scale)
        self.project = nn.Sequential(
            nn.Conv2d(
                in_channels * self.expansion, out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            if not lucid.random.bernoulli(self.p).item():
                return x

        x_shortcut = x
        x_residual = self.residual(x)
        x_se = self.se(x_residual)

        x = x_se * x_residual
        x = self.project(x)

        if self.shortcut:
            x = x_shortcut + x
        return x


class _SepConv(_MBConv):
    expansion: ClassVar[int] = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        se_scale: int = 4,
        p: float = 0.5,
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, se_scale, p)

        self.residual = nn.Sequential(
            nn.Conv2d(
                in_channels * self.expansion,
                in_channels * self.expansion,
                kernel_size=kernel_size,
                padding="same",
                groups=in_channels * self.expansion,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * self.expansion, momentum=0.99, eps=1e-3),
            nn.Swish(),
        )


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

        depth = depth_coef
        width = width_coef

        channels = [int(ch * width) for ch in channels]
        repeats = [int(rep * depth) for rep in repeats]

        if stochastic_depth:
            self.p = p
            self.step = 0.5 / (sum(repeats - 1))
        else:
            self.p = 1.0
            self.step = 0.0

        self.upsample = ...  # NOTE: need to implement `Upsample`

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3),
        )

        # TODO: Make remaining stages using `_make_block()`

    def _make_block(
        self,
        block: Type[nn.Module],
        repeats: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        se_scale: int,
    ) -> nn.Sequential:
        strides = [stride] + [1] * (repeats - 1)
        layers = []
        for stride in strides:
            layers.append(
                block(in_channels, out_channels, kernel_size, stride, se_scale, self.p)
            )
            in_channels = out_channels
            self.p -= self.step

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        NotImplemented


@register_model
def efficientnet_b0(num_classes: int = 1000, **kwargs) -> EfficientNet: ...


@register_model
def efficientnet_b1(num_classes: int = 1000, **kwargs) -> EfficientNet: ...


@register_model
def efficientnet_b2(num_classes: int = 1000, **kwargs) -> EfficientNet: ...


@register_model
def efficientnet_b3(num_classes: int = 1000, **kwargs) -> EfficientNet: ...


@register_model
def efficientnet_b4(num_classes: int = 1000, **kwargs) -> EfficientNet: ...


@register_model
def efficientnet_b5(num_classes: int = 1000, **kwargs) -> EfficientNet: ...


@register_model
def efficientnet_b6(num_classes: int = 1000, **kwargs) -> EfficientNet: ...


@register_model
def efficientnet_b7(num_classes: int = 1000, **kwargs) -> EfficientNet: ...
