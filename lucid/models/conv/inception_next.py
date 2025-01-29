from typing import Type
from functools import partial

import lucid
import lucid.nn as nn

from lucid import register_model
from lucid._tensor import Tensor


__all__ = [
    "InceptionNeXt",
    "inception_next_atto",
    "inception_next_tiny",
    "inception_next_small",
    "inception_next_base",
]


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
    def __init__(
        self,
        num_classes: int = 1000,
        depths: list[int] = [3, 3, 9, 3],
        dims: list[int] = [96, 192, 384, 768],
        token_mixers: Type[nn.Module] = nn.Identity,
        mlp_ratios: list[int] = [4, 4, 4, 3],
        head_fn: Type[nn.Module] = _MLPHead,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ls_init_value: float = 1e-6,
    ) -> None:
        super().__init__()
        num_stage = len(depths)
        if not isinstance(token_mixers, (list, tuple)):
            token_mixers = [token_mixers] * num_stage

        if not isinstance(mlp_ratios, (list, tuple)):
            mlp_ratios = [mlp_ratios] * num_stage

        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4), nn.BatchNorm2d(dims[0])
        )

        self.stages = None
        dp_rates = [x.item() for x in lucid.linspace(0, drop_path_rate, sum(depths))]
        stages = []
        prev_channels = dims[0]

        for i in range(num_stage):
            out_channels = dims[i]
            stage = _IncepNeXtStage(
                prev_channels,
                out_channels,
                ds_stride=2 if i > 0 else 1,
                depth=depths[i],
                drop_path_rates=dp_rates[i],
                ls_init_value=ls_init_value,
                token_mixer=token_mixers[i],
                norm_layer=nn.BatchNorm2d,
                mlp_ratio=mlp_ratios[i],
            )
            stages.append(stage)
            prev_channels = out_channels

        self.stages = nn.Sequential(*stages)

        self.num_features = prev_channels
        self.head = head_fn(self.num_features, num_classes, drop=drop_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)

        return x


@register_model
def inception_next_atto(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    return InceptionNeXt(
        num_classes,
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        token_mixers=partial(_InceptionDWConv2d, band_kernel_size=9),
        **kwargs
    )


@register_model
def inception_next_tiny(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    return InceptionNeXt(
        num_classes,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        token_mixers=_InceptionDWConv2d,
        **kwargs
    )


@register_model
def inception_next_small(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    return InceptionNeXt(
        num_classes,
        depths=[3, 3, 27, 3],
        dims=[96, 192, 384, 768],
        token_mixers=_InceptionDWConv2d,
        **kwargs
    )


@register_model
def inception_next_base(num_classes: int = 1000, **kwargs) -> InceptionNeXt:
    return InceptionNeXt(
        num_classes,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        token_mixers=_InceptionDWConv2d,
        **kwargs
    )
