from typing import override

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["ConvNeXt"]


class _Block(nn.Module):
    def __init__(
        self, in_channels: int, drop_path: float = 0.0, layer_scale_init: float = 1e-6
    ) -> None:
        super().__init__()

        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, padding=3, groups=in_channels
        )
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)

        self.pwconv1 = nn.Linear(in_channels, 4 * in_channels)
        self.gelu = nn.GELU()
        self.pwconv2 = nn.Linear(4 * in_channels, in_channels)

        self.gamma = (
            nn.Parameter(layer_scale_init * lucid.ones(in_channels))
            if layer_scale_init > 0
            else None
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input_ = x
        x = self.dwconv(x)
        x = x.transpose((0, 2, 3, 1))
        x = self.norm(x)

        n, h, w, _ = x.shape
        x = self.pwconv1(x.reshape(n * h * w, -1))
        x = self.gelu(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            flat_gamma = lucid.repeat(self.gamma, x.shape[0], axis=0)
            x = flat_gamma * x.reshape(-1)

        x = x.reshape(n, -1, h, w)
        x = input_ + self.drop_path(x)
        return x


class _ChannelsFisrtLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__(normalized_shape, eps, elementwise_affine, bias)

    @override
    def forward(self, x: Tensor) -> Tensor:
        x = x.transpose((0, 2, 3, 1))
        out = super().forward(x)
        out = out.transpose((0, 3, 1, 2))
        return out


class ConvNeXt(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
        head_init_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=4, stride=4),
            _ChannelsFisrtLayerNorm(dims[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample = nn.Sequential(
                _ChannelsFisrtLayerNorm(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in lucid.linspace(0, drop_path, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(...)

            # Begin from here
            # NOTE: refer to Meta's authentic ConvNeXt imeplementation;
