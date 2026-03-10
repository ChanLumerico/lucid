from dataclasses import dataclass
from functools import partial
from typing import Type

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "InceptionNeXt",
    "InceptionNeXtConfig",
    "inception_next_atto",
    "inception_next_tiny",
    "inception_next_small",
    "inception_next_base",
]


def _normalize_positive_int_sequence(
    values: tuple[int, ...] | list[int],
    name: str,
) -> tuple[int, ...]:
    normalized = tuple(values)
    if len(normalized) == 0:
        raise ValueError(f"{name} must contain at least one stage value")
    if any(not isinstance(value, int) or value <= 0 for value in normalized):
        raise ValueError(f"{name} values must be positive integers")
    return normalized


@dataclass
class InceptionNeXtConfig:
    num_classes: int = 1000
    depths: tuple[int, ...] | list[int] = (3, 3, 9, 3)
    dims: tuple[int, ...] | list[int] = (96, 192, 384, 768)
    token_mixers: object | None = None
    mlp_ratios: int | tuple[int, ...] | list[int] = (4, 4, 4, 3)
    head_fn: object | None = None
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    ls_init_value: float = 1e-6

    def __post_init__(self) -> None:
        self.depths = _normalize_positive_int_sequence(self.depths, "depths")
        self.dims = _normalize_positive_int_sequence(self.dims, "dims")
        num_stage = len(self.depths)

        if len(self.dims) != num_stage:
            raise ValueError("dims must have the same number of stages as depths")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")

        if self.token_mixers is None:
            self.token_mixers = _InceptionDWConv2d

        if isinstance(self.token_mixers, (list, tuple)):
            if len(self.token_mixers) != num_stage:
                raise ValueError(
                    "token_mixers must have the same number of stages as depths"
                )
            if any(not callable(token_mixer) for token_mixer in self.token_mixers):
                raise TypeError("token_mixers entries must be callable")
            self.token_mixers = tuple(self.token_mixers)
        elif callable(self.token_mixers):
            self.token_mixers = tuple(self.token_mixers for _ in range(num_stage))
        else:
            raise TypeError("token_mixers must be callable or a sequence of callables")

        if isinstance(self.mlp_ratios, (list, tuple)):
            if len(self.mlp_ratios) != num_stage:
                raise ValueError(
                    "mlp_ratios must have the same number of stages as depths"
                )
            if any(
                not isinstance(mlp_ratio, int) or mlp_ratio <= 0
                for mlp_ratio in self.mlp_ratios
            ):
                raise ValueError("mlp_ratios values must be positive integers")
            self.mlp_ratios = tuple(self.mlp_ratios)
        elif isinstance(self.mlp_ratios, int) and self.mlp_ratios > 0:
            self.mlp_ratios = tuple(self.mlp_ratios for _ in range(num_stage))
        else:
            raise ValueError("mlp_ratios must be a positive integer or sequence of positive integers")

        if self.head_fn is None:
            self.head_fn = _MLPHead
        if not callable(self.head_fn):
            raise TypeError("head_fn must be callable")
        if self.drop_rate < 0 or self.drop_rate >= 1:
            raise ValueError("drop_rate must be in the range [0, 1)")
        if self.drop_path_rate < 0 or self.drop_path_rate > 1:
            raise ValueError("drop_path_rate must be in the range [0, 1]")
        if self.ls_init_value < 0:
            raise ValueError("ls_init_value must be greater than or equal to 0")


class _InceptionDWConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        square_kernel_size: int = 3,
        band_kernel_size: int = 11,
        branch_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        gc = int(in_channels * branch_ratio)

        self.dwconv_hw = nn.Conv2d(
            gc, gc, kernel_size=square_kernel_size, padding="same", groups=gc
        )
        self.dwconv_w = nn.Conv2d(
            gc, gc, kernel_size=(1, band_kernel_size), padding="same", groups=gc
        )
        self.dwconv_h = nn.Conv2d(
            gc, gc, kernel_size=(band_kernel_size, 1), padding="same", groups=gc
        )
        self.split_indices = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x: Tensor) -> Tensor:
        x_id, x_hw, x_w, x_h = x.split(self.split_indices, axis=1)
        return lucid.concatenate(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            axis=1,
        )


class _ConvMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        bias: bool = True,
        act_layer: Type[nn.Module] = nn.ReLU,
        norm_layer: Type[nn.Module] | None = None,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1, bias=bias)
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()

        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.norm(self.fc1(x)))
        x = self.fc2(self.drop(x))

        return x


class _MLPHead(nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int = 1000,
        mlp_ratio: int = 3,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        hidden_features = int(mlp_ratio * dim)

        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.norm = norm_layer(hidden_features)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, num_classes, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.norm(self.act(self.fc1(x)))
        x = self.fc2(self.drop(x))

        return x


class _IncepNeXtBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        token_mixer: Type[nn.Module] = nn.Identity,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        mlp_layer: Type[nn.Module] = _ConvMLP,
        mlp_ratio: int = 4,
        act_layer: Type[nn.Module] = nn.GELU,
        ls_init_value: float = 1e-6,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.token_mixer = token_mixer(dim)
        self.norm = norm_layer(dim)
        self.mlp = mlp_layer(dim, int(mlp_ratio * dim), act_layer=act_layer)

        self.gamma = (
            nn.Parameter(ls_init_value * lucid.ones(dim)) if ls_init_value else None
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        shortcut = x
        x = self.token_mixer(x)
        x = self.norm(x)
        x = self.mlp(x)

        if self.gamma is not None:
            x *= self.gamma.reshape(1, -1, 1, 1)

        x = self.drop_path(x) + shortcut
        return x


class _IncepNeXtStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ds_stride: int = 2,
        depth: int = 2,
        drop_path_rates: list[float] | None = None,
        ls_init_value: float = 0.0,
        token_mixer: Type[nn.Module] = nn.Identity,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.BatchNorm2d,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        if ds_stride > 1:
            self.downsample = nn.Sequential(
                norm_layer(in_channels),
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=ds_stride,
                    stride=ds_stride,
                ),
            )
        else:
            self.downsample = nn.Identity()

        drop_path_rates = drop_path_rates or [0.0] * depth
        stage_blocks = []

        for i in range(depth):
            block = _IncepNeXtBlock(
                out_channels,
                token_mixer,
                norm_layer,
                mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                ls_init_value=ls_init_value,
                drop_path=drop_path_rates[i],
            )
            stage_blocks.append(block)
            in_channels = out_channels

        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class InceptionNeXt(nn.Module):
    def __init__(self, config: InceptionNeXtConfig) -> None:
        super().__init__()
        self.config = config
        num_stage = len(config.depths)

        self.num_classes = config.num_classes
        self.drop_rate = config.drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(3, config.dims[0], kernel_size=4, stride=4),
            nn.BatchNorm2d(config.dims[0]),
        )

        self.stages = None
        dp_rates = [
            x.item()
            for x in lucid.linspace(0, config.drop_path_rate, sum(config.depths))
        ]
        stages = []
        prev_channels = config.dims[0]
        cur = 0

        for i in range(num_stage):
            out_channels = config.dims[i]
            stage = _IncepNeXtStage(
                prev_channels,
                out_channels,
                ds_stride=2 if i > 0 else 1,
                depth=config.depths[i],
                drop_path_rates=dp_rates[cur : cur + config.depths[i]],
                ls_init_value=config.ls_init_value,
                token_mixer=config.token_mixers[i],
                norm_layer=nn.BatchNorm2d,
                mlp_ratio=config.mlp_ratios[i],
            )
            stages.append(stage)
            prev_channels = out_channels
            cur += config.depths[i]

        self.stages = nn.Sequential(*stages)

        self.num_features = prev_channels
        self.head = config.head_fn(
            self.num_features,
            config.num_classes,
            drop=config.drop_rate,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)

        return x


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_inception_next_config(
    *,
    num_classes: int,
    depths: tuple[int, ...],
    dims: tuple[int, ...],
    token_mixers: object,
    **kwargs,
) -> InceptionNeXtConfig:
    return InceptionNeXtConfig(
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        token_mixers=token_mixers,
        **kwargs,
    )


@register_model
def inception_next_atto(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims", "token_mixers"},
        "factory variants do not allow overriding preset depths, dims, or token_mixers",
    )
    config = _build_inception_next_config(
        num_classes=num_classes,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        token_mixers=partial(_InceptionDWConv2d, band_kernel_size=9),
        **kwargs,
    )
    return InceptionNeXt(config)


@register_model
def inception_next_tiny(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims", "token_mixers"},
        "factory variants do not allow overriding preset depths, dims, or token_mixers",
    )
    config = _build_inception_next_config(
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        token_mixers=_InceptionDWConv2d,
        **kwargs,
    )
    return InceptionNeXt(config)


@register_model
def inception_next_small(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims", "token_mixers"},
        "factory variants do not allow overriding preset depths, dims, or token_mixers",
    )
    config = _build_inception_next_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(96, 192, 384, 768),
        token_mixers=_InceptionDWConv2d,
        **kwargs,
    )
    return InceptionNeXt(config)


@register_model
def inception_next_base(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "dims", "token_mixers"},
        "factory variants do not allow overriding preset depths, dims, or token_mixers",
    )
    config = _build_inception_next_config(
        num_classes=num_classes,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        token_mixers=_InceptionDWConv2d,
        **kwargs,
    )
    return InceptionNeXt(config)
