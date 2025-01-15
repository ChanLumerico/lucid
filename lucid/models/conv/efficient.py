from typing import ClassVar

import lucid.nn as nn

import lucid
from lucid import register_model
from lucid._tensor import Tensor


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


class MBConv(nn.Module):
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
            ...  # TODO: implement `lucid.random.bernoulli()`
