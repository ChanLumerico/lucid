from typing import override, ClassVar, Type

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "ConvNeXt",
    "ConvNeXt_V2",
    "convnext_tiny",
    "convnext_small",
    "convnext_base",
    "convnext_large",
    "convnext_xlarge",
    "convnext_v2_atto",
    "convnext_v2_femto",
    "convnext_v2_pico",
    "convnext_v2_nano",
    "convnext_v2_tiny",
    "convnext_v2_base",
    "convnext_v2_large",
    "convnext_v2_huge",
]


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
    base_block: ClassVar[Type[nn.Module]] = _Block

    def __init__(
        self,
        num_classes: int = 1000,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path: float = 0.0,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
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
            stage = nn.Sequential(
                *[
                    self.base_block(dims[i], dp_rates[cur + j], layer_scale_init)
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.norm(x)
        x = self.head(x)

        return x


class _Block_V2(nn.Module):
    def __init__(self, channels: int, drop_path: float = 0.0, *args) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels)

        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.gelu = nn.GELU()
        self.grn = nn.GlobalResponseNorm(4 * channels)

        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        input_ = x
        x = self.dwconv(x).transpose((0, 2, 3, 1))
        x = self.norm(x)

        n, h, w, _ = x.shape
        x = self.pwconv1(x.reshape(n * h * w, -1))
        x = self.gelu(x)
        x = self.grn(x.reshape(n, -1, h, w))
        x = self.pwconv2(x.reshape(n * h * w, -1))

        x = x.reshape(n, -1, h, w)
        x = input_ + self.drop_path(x)

        return x


class ConvNeXt_V2(ConvNeXt):
    base_block: ClassVar[Type[nn.Module]] = _Block_V2

    def __init__(
        self,
        num_classes: int = 1000,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        drop_path: float = 0.0,
    ) -> None:
        super().__init__(num_classes, depths, dims, drop_path)


@register_model
def convnext_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    depths = [3, 3, 9, 3]
    dims = [96, 192, 384, 768]
    return ConvNeXt(num_classes, depths, dims, **kwargs)


@register_model
def convnext_small(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    depths = [3, 3, 27, 3]
    dims = [96, 192, 364, 768]
    return ConvNeXt(num_classes, depths, dims, **kwargs)


@register_model
def convnext_base(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    depths = [3, 3, 27, 3]
    dims = [128, 256, 512, 1024]
    return ConvNeXt(num_classes, depths, dims, **kwargs)


@register_model
def convnext_large(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    depths = [3, 3, 27, 3]
    dims = [192, 384, 768, 1536]
    return ConvNeXt(num_classes, depths, dims, **kwargs)


@register_model
def convnext_xlarge(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    depths = [3, 3, 27, 3]
    dims = [256, 512, 1024, 2048]
    return ConvNeXt(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_atto(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [2, 2, 6, 2]
    dims = [40, 80, 160, 320]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_femto(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [2, 2, 6, 2]
    dims = [48, 96, 192, 384]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_pico(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [2, 2, 6, 2]
    dims = [64, 128, 256, 512]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_nano(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [2, 2, 8, 2]
    dims = [80, 160, 320, 640]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [3, 3, 9, 3]
    dims = [96, 192, 384, 768]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_base(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [3, 3, 27, 3]
    dims = [128, 256, 512, 1024]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_large(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [3, 3, 27, 3]
    dims = [192, 384, 768, 1536]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)


@register_model
def convnext_v2_huge(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    depths = [3, 3, 27, 3]
    dims = [352, 704, 1408, 2816]
    return ConvNeXt_V2(num_classes, depths, dims, **kwargs)
