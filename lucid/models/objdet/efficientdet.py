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

    def _norm_weight(self, weight: Tensor) -> Tensor:
        return weight / (weight.sum(axis=0) + self.eps)

    def _forward_up(self, feats: tuple[Tensor]) -> tuple[Tensor]:
        p3_in, p4_in, p5_in, p6_in, p7_in = feats

        w1_p6_up = self._norm_weight(self.relus["6_w1"](self.weights["6_w1"]))
        p6_up_in = w1_p6_up[0] * p6_in + w1_p6_up[1] * self.ups["6"](p7_in)
        p6_up = self.convs["6_up"](p6_up_in)

        w1_p5_up = self._norm_weight(self.relus["5_w1"](self.weights["5_w1"]))
        p5_up_in = w1_p5_up[0] * p5_in + w1_p5_up[1] * self.ups["5"](p6_up)
        p5_up = self.convs["5_up"](p5_up_in)

        w1_p4_up = self._norm_weight(self.relus["4_w1"](self.weights["4_w1"]))
        p4_up_in = w1_p4_up[0] * p4_in + w1_p4_up[1] * self.ups["4"](p5_up)
        p4_up = self.convs["4_up"](p4_up_in)

        w1_p3_up = self._norm_weight(self.relus["3_w1"](self.weights["3_w1"]))
        p3_up_in = w1_p3_up[0] * p3_in + w1_p3_up[1] * self.ups["3"](p4_up)
        p3_out = self.convs["3_up"](p3_up_in)

        return p3_out, p4_up, p5_up, p6_up

    def _forward_down(
        self, feats: tuple[Tensor], up_feats: tuple[Tensor]
    ) -> tuple[Tensor]:
        _, p4_in, p5_in, p6_in, p7_in = feats
        p3_out, p4_up, p5_up, p6_up = up_feats

        w2_p4_down = self._norm_weight(self.relus["4_w2"](self.weights["4_w2"]))
        p4_down_in = (
            w2_p4_down[0] * p4_in
            + w2_p4_down[1] * p4_up
            + w2_p4_down[2] * self.downs["4"](p3_out)
        )
        p4_out = self.convs["4_down"](p4_down_in)

        w2_p5_down = self._norm_weight(self.relus["5_w2"](self.weights["5_w2"]))
        p5_down_in = (
            w2_p5_down[0] * p5_in
            + w2_p5_down[1] * p5_up
            + w2_p5_down[2] * self.downs["5"](p4_out)
        )
        p5_out = self.convs["5_down"](p5_down_in)

        w2_p6_down = self._norm_weight(self.relus["6_w2"](self.weights["6_w2"]))
        p6_down_in = (
            w2_p6_down[0] * p6_in
            + w2_p6_down[1] * p6_up
            + w2_p6_down[2] * self.downs["6"](p5_out)
        )
        p6_out = self.convs["6_down"](p6_down_in)

        w2_p7_down = self._norm_weight(self.relus["7_w2"](self.weights["7_w2"]))
        p7_down_in = w2_p7_down[0] * p7_in + w2_p7_down[1] * self.downs["7"](p6_out)
        p7_out = self.convs["7_down"](p7_down_in)

        return p3_out, p4_out, p5_out, p6_out, p7_out

    def forward(self, feats: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        up_feats = self._forward_up(feats)
        down_feats = self._forward_down(feats, up_feats)

        return down_feats


class _BBoxRegresor(nn.Module):
    def __init__(self, in_channels: int, num_anchors: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU)

        self.layers = nn.Sequential(*layers)
        self.header = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        x = self.header(x)

        out = x.transpose((0, 2, 3, 1))
        return out.reshape(out.shape[0], -1, 4)
