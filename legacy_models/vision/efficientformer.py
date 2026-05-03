from dataclasses import dataclass
from typing import Callable
from functools import partial
import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "EfficientFormer",
    "EfficientFormerConfig",
    "efficientformer_l1",
    "efficientformer_l3",
    "efficientformer_l7",
]


def _to_2_tuple(val: int | float) -> tuple:
    return (val, val)


@dataclass
class EfficientFormerConfig:
    depths: tuple[int, ...] | list[int]
    embed_dims: tuple[int, ...] | list[int]
    in_channels: int = 3
    num_classes: int = 1000
    global_pool: bool = True
    downsamples: tuple[bool, ...] | list[bool] | None = None
    num_vit: int = 0
    mlp_ratios: float = 4.0
    pool_size: int = 3
    layer_scale_init_value: float = 1e-5
    act_layer: type[nn.Module] = nn.GELU
    norm_layer: type[nn.Module] = nn.BatchNorm2d
    norm_layer_cl: type[nn.Module] = nn.LayerNorm
    drop_rate: float = 0.0
    proj_drop_rate: float = 0.0
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        self.depths = tuple(self.depths)
        self.embed_dims = tuple(self.embed_dims)
        if self.downsamples is not None:
            self.downsamples = tuple(self.downsamples)

        if len(self.depths) == 0 or any(depth <= 0 for depth in self.depths):
            raise ValueError("depths must contain at least one positive integer")
        if len(self.embed_dims) != len(self.depths):
            raise ValueError("embed_dims must provide one positive width per stage")
        if any(dim <= 0 for dim in self.embed_dims):
            raise ValueError("embed_dims must contain only positive integers")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.num_classes < 0:
            raise ValueError("num_classes must be greater than or equal to 0")
        if self.downsamples is not None and len(self.downsamples) != len(self.depths):
            raise ValueError("downsamples must match the number of stages")
        if self.num_vit < 0:
            raise ValueError("num_vit must be greater than or equal to 0")
        if self.mlp_ratios <= 0:
            raise ValueError("mlp_ratios must be greater than 0")
        if self.pool_size <= 0:
            raise ValueError("pool_size must be greater than 0")
        if self.layer_scale_init_value < 0:
            raise ValueError("layer_scale_init_value must be non-negative")
        if self.drop_rate < 0 or self.drop_rate >= 1:
            raise ValueError("drop_rate must be in the range [0, 1)")
        if self.proj_drop_rate < 0 or self.proj_drop_rate >= 1:
            raise ValueError("proj_drop_rate must be in the range [0, 1)")
        if self.drop_path_rate < 0 or self.drop_path_rate >= 1:
            raise ValueError("drop_path_rate must be in the range [0, 1)")


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] | None = None,
        bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = _to_2_tuple(bias)
        drop_probs = _to_2_tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(self.norm(x)))
        return x


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int = 384,
        key_dim: int = 32,
        num_heads: int = 8,
        attn_ratio: float = 4.0,
        resolution: int = 7,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.key_attn_dim = key_dim * num_heads
        self.val_dim = int(attn_ratio * key_dim)
        self.val_attn_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio

        self.qkv = nn.Linear(dim, self.key_attn_dim * 2 + self.val_attn_dim)
        self.proj = nn.Linear(self.val_attn_dim, dim)

        resolution = _to_2_tuple(resolution)
        y, x = lucid.meshgrid(
            lucid.arange(resolution[0]), lucid.arange(resolution[1]), indexing="ij"
        )
        pos = lucid.stack([y, x]).flatten(start_axis=1)
        rel_pos = lucid.abs(pos[..., :, None] - pos[..., None, :])
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]

        self.attention_biases = nn.Parameter(
            lucid.zeros(num_heads, resolution[0] * resolution[1])
        )
        self.attention_bias_idxs: nn.Buffer
        self.register_buffer("attention_bias_idxs", rel_pos.astype(lucid.Int))

    def forward(self, x: Tensor) -> Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, self.num_heads, -1).transpose((0, 2, 1, 3))
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.val_dim], axis=3)

        attn = (q @ k.mT) * self.scale
        attn += self.attention_biases[:, self.attention_bias_idxs]
        attn = F.softmax(attn, axis=-1)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, self.val_attn_dim)
        x = self.proj(x)

        return x


class _Stem(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: type[nn.Module] = nn.ReLU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.stride = 4

        self.add_module(
            "conv1",
            nn.Conv2d(
                in_channels, out_channels // 2, kernel_size=3, stride=2, padding=1
            ),
        )
        self.add_module("norm1", norm_layer(out_channels // 2))
        self.add_module("act1", act_layer())
        self.add_module(
            "conv2",
            nn.Conv2d(
                out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1
            ),
        )
        self.add_module("norm2", norm_layer(out_channels))
        self.add_module("act2", act_layer())


class _Downsample(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int | None = None,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = norm_layer(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


class _Flatten(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_axis=2).swapaxes(1, 2)
        return x


class _Pooling(nn.Module):
    def __init__(self, pool_size: int = 3) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x) -> None:
        return self.pool(x) - x


class _ConvMLPNorm(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.norm1 = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.act = act_layer()

        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.norm2 = (
            norm_layer(out_features) if norm_layer is not None else nn.Identity()
        )
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.act(self.norm1(self.fc1(x)))
        x = self.drop(x)
        x = self.norm2(self.fc2(x))
        x = self.drop(x)

        return x


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(lucid.full(dim, init_value))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _LayerScale2d(_LayerScale):
    def forward(self, x: Tensor) -> Tensor:
        gamma = self.gamma.reshape(1, -1, 1, 1)
        return x * gamma


class _MetaBlock1d(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = _Attention(dim)
        self.norm2 = norm_layer(dim)
        self.mlp = _MLP(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ls1 = _LayerScale(dim, layer_scale_init_value)
        self.ls2 = _LayerScale(dim, layer_scale_init_value)

    def forward(self, x: Tensor) -> Tensor:
        x += self.drop_path(self.ls1(self.token_mixer(self.norm1(x))))
        x += self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class _MetaBlock2d(nn.Module):
    def __init__(
        self,
        dim: int,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super().__init__()
        self.token_mixer = _Pooling(pool_size)
        self.ls1 = _LayerScale2d(dim, layer_scale_init_value)
        self.drop_path1 = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = _ConvMLPNorm(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=proj_drop,
        )
        self.ls2 = _LayerScale2d(dim, layer_scale_init_value)
        self.drop_path2 = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x += self.drop_path1(self.ls1(self.token_mixer(x)))
        x += self.drop_path2(self.ls2(self.mlp(x)))
        return x


class _EfficientFormerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        downsample: bool = True,
        num_vit: int = 1,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.BatchNorm2d,
        norm_layer_cl: type[nn.Module] = nn.LayerNorm,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-5,
    ) -> None:
        super().__init__()
        if downsample:
            self.downsample = _Downsample(dim, dim_out, norm_layer=norm_layer)
            dim = dim_out
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        if num_vit and num_vit >= depth:
            blocks.append(_Flatten())

        for block_idx in range(depth):
            remain_idx = depth - block_idx - 1
            if num_vit and num_vit > remain_idx:
                blocks.append(
                    _MetaBlock1d(
                        dim,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer_cl,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
            else:
                blocks.append(
                    _MetaBlock2d(
                        dim,
                        pool_size=pool_size,
                        mlp_ratio=mlp_ratio,
                        act_layer=act_layer,
                        norm_layer=norm_layer,
                        proj_drop=proj_drop,
                        drop_path=drop_path[block_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                if num_vit and num_vit == remain_idx:
                    blocks.append(_Flatten())

        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class EfficientFormer(nn.Module):
    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.global_pool = config.global_pool

        self.stem = _Stem(
            config.in_channels, config.embed_dims[0], norm_layer=config.norm_layer
        )
        prev_dim = config.embed_dims[0]

        self.num_stages = len(config.depths)
        last_stage = self.num_stages - 1
        dpr = [
            x.tolist()
            for x in lucid.linspace(0, config.drop_path_rate, sum(config.depths)).split(
                config.depths
            )
        ]
        downsamples = config.downsamples or (False,) + (True,) * (self.num_stages - 1)

        stages = []
        for i in range(self.num_stages):
            stage = _EfficientFormerStage(
                prev_dim,
                config.embed_dims[i],
                config.depths[i],
                downsample=downsamples[i],
                num_vit=config.num_vit if i == last_stage else 0,
                pool_size=config.pool_size,
                mlp_ratio=config.mlp_ratios,
                act_layer=config.act_layer,
                norm_layer_cl=config.norm_layer_cl,
                norm_layer=config.norm_layer,
                proj_drop=config.proj_drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=config.layer_scale_init_value,
            )
            prev_dim = config.embed_dims[i]
            stages.append(stage)

        self.stages = nn.Sequential(*stages)

        self.num_features = config.embed_dims[-1]
        self.norm = config.norm_layer_cl(self.num_features)
        self.head_drop = nn.Dropout(config.drop_rate)
        self.head = (
            nn.Linear(self.num_features, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant(m.bias, 0.0)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: Tensor, pre_logits: bool = False) -> Tensor:
        if self.global_pool:
            x = x.mean(axis=1)

        x = self.head_drop(x)
        if pre_logits:
            return x

        x = self.head(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


EfficientFormer_width = {
    "l1": (48, 96, 224, 448),
    "l3": (64, 128, 320, 512),
    "l7": (96, 192, 384, 768),
}

EfficientFormer_depth = {
    "l1": (3, 2, 6, 4),
    "l3": (4, 4, 12, 6),
    "l7": (6, 6, 18, 8),
}


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_efficientformer_config(
    *,
    num_classes: int,
    depths: tuple[int, ...] | list[int],
    embed_dims: tuple[int, ...] | list[int],
    num_vit: int,
    **kwargs,
) -> EfficientFormerConfig:
    params = {
        "num_classes": num_classes,
        "depths": depths,
        "embed_dims": embed_dims,
        "num_vit": num_vit,
    }
    params.update(kwargs)
    return EfficientFormerConfig(**params)


@register_model
def efficientformer_l1(num_classes: int = 1000, **kwargs) -> EfficientFormer:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "embed_dims", "num_vit"},
        "factory variants do not allow overriding preset depths, embed_dims, or num_vit",
    )
    config = _build_efficientformer_config(
        depths=EfficientFormer_depth["l1"],
        embed_dims=EfficientFormer_width["l1"],
        num_vit=1,
        num_classes=num_classes,
        **kwargs,
    )
    return EfficientFormer(config)


@register_model
def efficientformer_l3(num_classes: int = 1000, **kwargs) -> EfficientFormer:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "embed_dims", "num_vit"},
        "factory variants do not allow overriding preset depths, embed_dims, or num_vit",
    )
    config = _build_efficientformer_config(
        depths=EfficientFormer_depth["l3"],
        embed_dims=EfficientFormer_width["l3"],
        num_vit=4,
        num_classes=num_classes,
        **kwargs,
    )
    return EfficientFormer(config)


@register_model
def efficientformer_l7(num_classes: int = 1000, **kwargs) -> EfficientFormer:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"depths", "embed_dims", "num_vit"},
        "factory variants do not allow overriding preset depths, embed_dims, or num_vit",
    )
    config = _build_efficientformer_config(
        depths=EfficientFormer_depth["l7"],
        embed_dims=EfficientFormer_width["l7"],
        num_vit=8,
        num_classes=num_classes,
        **kwargs,
    )
    return EfficientFormer(config)
