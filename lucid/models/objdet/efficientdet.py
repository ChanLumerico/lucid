from lucid import register_model

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


class _ConvBlock(nn.Module):
    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                num_channels,
                num_channels,
                kernel_size=3,
                padding=1,
                groups=num_channels,
            ),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels, momentum=0.9997, eps=4e-5),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _BiFPN(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.eps = eps
        self.convs = nn.ModuleDict(
            {
                **{f"{i}_up": _ConvBlock(num_channels) for i in range(3, 7)},
                **{f"{i}_down": _ConvBlock(num_channels) for i in range(4, 8)},
            }
        )
        self.ups = nn.ModuleDict(
            {f"{i}": nn.Upsample(scale_factor=2, mode="nearest") for i in range(3, 7)}
        )
        self.downs = nn.ModuleDict(
            {f"{i}": nn.AvgPool2d(kernel_size=2) for i in range(4, 8)}
        )

        self.weights = nn.ModuleDict(
            {
                **{f"{i}_w1": nn.Parameter(lucid.ones(2)) for i in range(3, 7)},
                **{f"{i}_w2": nn.Parameter(lucid.ones(3)) for i in range(4, 8)},
            }
        )
        self.relus = nn.ModuleDict(
            {
                **{f"{i}_w1": nn.ReLU() for i in range(3, 7)},
                **{f"{i}_w2": nn.ReLU() for i in range(4, 8)},
            }
        )

    def norm_weight(self, weight: Tensor) -> Tensor:
        return weight / (weight.sum(axis=0) + self.eps)

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        p3_in, p4_in, p5_in, p6_in, p7_in = feats

        w_p6_up = self.norm_weight(self.relus["6_w1"](self.weights["6_w1"]))
        p6_up = self.convs["6_up"](
            w_p6_up[0] * p6_in + w_p6_up[1] * self.ups["6"](p7_in)
        )

        w_p5_up = self.norm_weight(self.relus["5_w1"](self.weights["5_w1"]))
        p5_up = self.convs["5_up"](
            w_p5_up[0] * p5_in + w_p5_up[1] * self.ups["5"](p6_up)
        )

        # TODO: Continue from here
        NotImplemented
