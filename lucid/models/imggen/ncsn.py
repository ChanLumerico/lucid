from typing import Sequence

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid._tensor import Tensor


class _CondInstanceNorm(nn.Module):
    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=eps)
        self.embed = nn.Embedding(num_classes, num_features * 2)

        nn.init.constant(self.embed.weight[:, :num_features], 1.0)
        nn.init.constant(self.embed.weight[:, num_features:], 0.0)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if y.dtype != lucid.Long:
            y = y.long()

        h = self.norm(x)
        gamma_beta = self.embed(y)
        gamma, beta = lucid.chunk(gamma_beta, 2, axis=1)

        gamma = gamma.reshape(-1, self.num_features, 1, 1)
        beta = beta.reshape(-1, self.num_features, 1, 1)

        return h * gamma + beta


class _Conv3x3(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int = 1, bias: bool = True
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _ResidualConvUnit(nn.Module):
    def __init__(self, channels: int, num_classes: int, dilation: int = 1) -> None:
        super().__init__()
        self.norm1 = _CondInstanceNorm(channels, num_classes)
        self.conv1 = _Conv3x3(channels, channels, dilation=dilation)

        self.norm2 = _CondInstanceNorm(channels, num_classes)
        self.conc2 = _Conv3x3(channels, channels, dilation=dilation)

        self.act = nn.ELU()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = self.conv1(self.act(self.norm1(x, y)))
        h = self.conv2(self.act(self.norm2(x, y)))

        return x + h


class _RCUBlock(nn.Module):
    def __init__(
        self, channels: int, num_classes: int, num_units: int = 2, diltation: int = 1
    ) -> None:
        super().__init__()
        self.units = nn.ModuleList(
            [
                _ResidualConvUnit(channels, num_classes, dilation=diltation)
                for _ in range(num_units)
            ]
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        h = x
        for unit in self.units:
            h = unit(h, y)
        return h


class _CondAdapter(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_classes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if in_channels == out_channels:
            self.norm = None
            self.conv = None
        else:
            self.norm = _CondInstanceNorm(in_channels, num_classes)
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0
            )
        self.act = nn.ELU()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.in_channels == self.out_channels:
            return x
        return self.conv(self.act(self.norm(x, y)))


class _MultiResFusion(nn.Module):
    def __init__(
        self, in_channels_arr: Sequence[int], out_channels: int, num_classes: int
    ) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.norms = nn.ModuleList(
            [_CondInstanceNorm(c, num_classes) for c in in_channels_arr]
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1)
                for c in in_channels_arr
            ]
        )
        self.act = nn.ELU()

    def forward(self, xs: Sequence[Tensor], y: Tensor) -> Tensor:
        if len(xs) != len(self.convs):
            raise ValueError(f"Expected {len(self.conv)} inputs, got {len(xs)}")

        target_h = max(x.shape[-2] for x in xs)
        target_w = max(x.shape[-1] for x in xs)
        fused = None

        for x, norm, conv in zip(xs, self.norms, self.convs):
            h = conv(self.act(norm(x, y)))
            if h.shape[-2:] != (target_h, target_w):
                h = F.interpolate(h, size=(target_h, target_w), mode="nearest")

            fused = h if fused is None else fused + h

        return fused


class _ChainedResPooling(nn.Module): ...


class _RefineBlock(nn.Module): ...


class NCSN(nn.Module): ...
