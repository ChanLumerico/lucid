from dataclasses import dataclass
from typing import ClassVar, Type, override

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "ConvNeXt",
    "ConvNeXtConfig",
    "ConvNeXt_V2",
    "ConvNeXtV2Config",
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


def _normalize_stage_spec(
    values: tuple[int, ...] | list[int],
    name: str,
) -> tuple[int, int, int, int]:
    normalized = tuple(values)
    if len(normalized) != 4:
        raise ValueError(f"{name} must contain exactly 4 stage values")
    if any(not isinstance(value, int) or value <= 0 for value in normalized):
        raise ValueError(f"{name} values must be positive integers")
    return normalized  # type: ignore[return-value]


@dataclass
class ConvNeXtConfig:
    num_classes: int = 1000
    depths: tuple[int, int, int, int] | list[int] = (3, 3, 9, 3)
    dims: tuple[int, int, int, int] | list[int] = (96, 192, 384, 768)
    drop_path: float = 0.0
    layer_scale_init: float = 1e-6

    def __post_init__(self) -> None:
        self.depths = _normalize_stage_spec(self.depths, "depths")
        self.dims = _normalize_stage_spec(self.dims, "dims")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.drop_path < 0 or self.drop_path > 1:
            raise ValueError("drop_path must be in the range [0, 1]")
        if self.layer_scale_init < 0:
            raise ValueError("layer_scale_init must be greater than or equal to 0")


@dataclass
class ConvNeXtV2Config:
    num_classes: int = 1000
    depths: tuple[int, int, int, int] | list[int] = (3, 3, 9, 3)
    dims: tuple[int, int, int, int] | list[int] = (96, 192, 384, 768)
    drop_path: float = 0.0

    def __post_init__(self) -> None:
        self.depths = _normalize_stage_spec(self.depths, "depths")
        self.dims = _normalize_stage_spec(self.dims, "dims")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.drop_path < 0 or self.drop_path > 1:
            raise ValueError("drop_path must be in the range [0, 1]")


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

    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__()
        self.config = config

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, config.dims[0], kernel_size=4, stride=4),
            _ChannelsFisrtLayerNorm(config.dims[0]),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample = nn.Sequential(
                _ChannelsFisrtLayerNorm(config.dims[i]),
                nn.Conv2d(config.dims[i], config.dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample)

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in lucid.linspace(0, config.drop_path, sum(config.depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    self.base_block(
                        config.dims[i],
                        dp_rates[cur + j],
                        config.layer_scale_init,
                    )
                    for j in range(config.depths[i])
                ]
            )
            self.stages.append(stage)
            cur += config.depths[i]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.norm = nn.LayerNorm(config.dims[-1])
        self.head = nn.Linear(config.dims[-1], config.num_classes)

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

    def __init__(self, config: ConvNeXtV2Config) -> None:
        super().__init__(
            ConvNeXtConfig(
                num_classes=config.num_classes,
                depths=config.depths,
                dims=config.dims,
                drop_path=config.drop_path,
            )
        )
        self.config = config


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_convnext_config(
    *,
    num_classes: int,
    depths: tuple[int, int, int, int],
    dims: tuple[int, int, int, int],
    **kwargs,
) -> ConvNeXtConfig:
    return ConvNeXtConfig(
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        **kwargs,
    )


def _build_convnext_v2_config(
    *,
    num_classes: int,
    depths: tuple[int, int, int, int],
    dims: tuple[int, int, int, int],
    **kwargs,
) -> ConvNeXtV2Config:
    return ConvNeXtV2Config(
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        **kwargs,
    )


@register_model
def convnext_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_config(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        **kwargs,
    )
    return ConvNeXt(config)


@register_model
def convnext_small(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 364, 768),
        **kwargs,
    )
    return ConvNeXt(config)


@register_model
def convnext_base(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        **kwargs,
    )
    return ConvNeXt(config)


@register_model
def convnext_large(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        **kwargs,
    )
    return ConvNeXt(config)


@register_model
def convnext_xlarge(num_classes: int = 1000, **kwargs) -> ConvNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(256, 512, 1024, 2048),
        **kwargs,
    )
    return ConvNeXt(config)


@register_model
def convnext_v2_atto(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        **kwargs,
    )
    return ConvNeXt_V2(config)


@register_model
def convnext_v2_femto(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(2, 2, 6, 2),
        dims=(48, 96, 192, 384),
        **kwargs,
    )
    return ConvNeXt_V2(config)


@register_model
def convnext_v2_pico(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 256, 512),
        **kwargs,
    )
    return ConvNeXt_V2(config)


@register_model
def convnext_v2_nano(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(2, 2, 8, 2),
        dims=(80, 160, 320, 640),
        **kwargs,
    )
    return ConvNeXt_V2(config)


@register_model
def convnext_v2_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        **kwargs,
    )
    return ConvNeXt_V2(config)


@register_model
def convnext_v2_base(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        **kwargs,
    )
    return ConvNeXt_V2(config)


@register_model
def convnext_v2_large(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(192, 384, 768, 1536),
        **kwargs,
    )
    return ConvNeXt_V2(config)


@register_model
def convnext_v2_huge(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims"},
        "factory variants do not allow overriding preset depths or dims",
    )
    config = _build_convnext_v2_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(352, 704, 1408, 2816),
        **kwargs,
    )
    return ConvNeXt_V2(config)
