from functools import partial
from dataclasses import dataclass
import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "PVT",
    "PVTConfig",
    "PVT_V2",
    "PVTV2Config",
    "pvt_tiny",
    "pvt_small",
    "pvt_medium",
    "pvt_large",
    "pvt_huge",
    "pvt_v2_b0",
    "pvt_v2_b1",
    "pvt_v2_b2",
    "pvt_v2_b2_li",
    "pvt_v2_b3",
    "pvt_v2_b4",
    "pvt_v2_b5",
]


def _to_2tuple(val: int | float) -> tuple[int | float, ...]:
    return (val, val)


def _conv_output_size(size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
    return (size + 2 * padding - kernel_size) // stride + 1


@dataclass
class PVTConfig:
    img_size: int = 224
    num_classes: int = 1000
    patch_size: int = 4
    in_channels: int = 3
    embed_dims: tuple[int, ...] | list[int] = (64, 128, 256, 512)
    num_heads: tuple[int, ...] | list[int] = (1, 2, 4, 8)
    mlp_ratios: tuple[float, ...] | list[float] = (4.0, 4.0, 4.0, 4.0)
    qkv_bias: bool = False
    qk_scale: float | None = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_layer: type[nn.Module] = nn.LayerNorm
    depths: tuple[int, ...] | list[int] = (3, 4, 6, 3)
    sr_ratios: tuple[int, ...] | list[int] = (8, 4, 2, 1)

    def __post_init__(self) -> None:
        self.embed_dims = tuple(self.embed_dims)
        self.num_heads = tuple(self.num_heads)
        self.mlp_ratios = tuple(self.mlp_ratios)
        self.depths = tuple(self.depths)
        self.sr_ratios = tuple(self.sr_ratios)

        if self.img_size <= 0:
            raise ValueError("img_size must be greater than 0")
        if self.num_classes < 0:
            raise ValueError("num_classes must be greater than or equal to 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if any(len(values) != 4 for values in (self.embed_dims, self.num_heads, self.mlp_ratios, self.depths, self.sr_ratios)):
            raise ValueError("embed_dims, num_heads, mlp_ratios, depths, and sr_ratios must each contain exactly 4 values")
        if any(value <= 0 for value in self.embed_dims):
            raise ValueError("embed_dims must contain positive integers")
        if any(value <= 0 for value in self.num_heads):
            raise ValueError("num_heads must contain positive integers")
        if any(value <= 0 for value in self.mlp_ratios):
            raise ValueError("mlp_ratios must contain positive values")
        if any(value <= 0 for value in self.depths):
            raise ValueError("depths must contain positive integers")
        if any(value <= 0 for value in self.sr_ratios):
            raise ValueError("sr_ratios must contain positive integers")
        for dim, heads in zip(self.embed_dims, self.num_heads):
            if dim % heads != 0:
                raise ValueError("embed_dims values must be divisible by num_heads")
        if self.drop_rate < 0 or self.drop_rate >= 1:
            raise ValueError("drop_rate must be in the range [0, 1)")
        if self.attn_drop_rate < 0 or self.attn_drop_rate >= 1:
            raise ValueError("attn_drop_rate must be in the range [0, 1)")
        if self.drop_path_rate < 0 or self.drop_path_rate >= 1:
            raise ValueError("drop_path_rate must be in the range [0, 1)")

        stage_size = _conv_output_size(self.img_size, self.patch_size, self.patch_size)
        for _ in range(3):
            stage_size = _conv_output_size(stage_size, 2, 2)
            if stage_size <= 0:
                raise ValueError(
                    "img_size and patch_size do not provide enough spatial resolution for 4 PVT stages"
                )


@dataclass
class PVTV2Config:
    img_size: int = 224
    patch_size: int = 7
    in_channels: int = 3
    num_classes: int = 1000
    embed_dims: tuple[int, ...] | list[int] = (64, 128, 256, 512)
    num_heads: tuple[int, ...] | list[int] = (1, 2, 4, 8)
    mlp_ratios: tuple[int, ...] | list[int] = (4, 4, 4, 4)
    qkv_bias: bool = False
    qk_scale: float | None = None
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    norm_layer: type[nn.Module] = nn.LayerNorm
    depths: tuple[int, ...] | list[int] = (3, 4, 6, 3)
    sr_ratios: tuple[int, ...] | list[int] = (8, 4, 2, 1)
    num_stages: int = 4
    linear: bool = False

    def __post_init__(self) -> None:
        self.embed_dims = tuple(self.embed_dims)
        self.num_heads = tuple(self.num_heads)
        self.mlp_ratios = tuple(self.mlp_ratios)
        self.depths = tuple(self.depths)
        self.sr_ratios = tuple(self.sr_ratios)

        if self.img_size <= 0:
            raise ValueError("img_size must be greater than 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")
        if self.patch_size <= 4:
            raise ValueError("patch_size must be greater than 4 for the first overlap patch embedding")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.num_classes < 0:
            raise ValueError("num_classes must be greater than or equal to 0")
        if self.num_stages <= 0:
            raise ValueError("num_stages must be greater than 0")
        if any(
            len(values) != self.num_stages
            for values in (self.embed_dims, self.num_heads, self.mlp_ratios, self.depths, self.sr_ratios)
        ):
            raise ValueError("embed_dims, num_heads, mlp_ratios, depths, and sr_ratios must each contain exactly num_stages values")
        if any(value <= 0 for value in self.embed_dims):
            raise ValueError("embed_dims must contain positive integers")
        if any(value <= 0 for value in self.num_heads):
            raise ValueError("num_heads must contain positive integers")
        if any(value <= 0 for value in self.mlp_ratios):
            raise ValueError("mlp_ratios must contain positive values")
        if any(value <= 0 for value in self.depths):
            raise ValueError("depths must contain positive integers")
        if any(value <= 0 for value in self.sr_ratios):
            raise ValueError("sr_ratios must contain positive integers")
        for dim, heads in zip(self.embed_dims, self.num_heads):
            if dim % heads != 0:
                raise ValueError("embed_dims values must be divisible by num_heads")
        if self.drop_rate < 0 or self.drop_rate >= 1:
            raise ValueError("drop_rate must be in the range [0, 1)")
        if self.attn_drop_rate < 0 or self.attn_drop_rate >= 1:
            raise ValueError("attn_drop_rate must be in the range [0, 1)")
        if self.drop_path_rate < 0 or self.drop_path_rate >= 1:
            raise ValueError("drop_path_rate must be in the range [0, 1)")

        stage_size = _conv_output_size(self.img_size, self.patch_size, 4, self.patch_size // 2)
        if stage_size <= 0:
            raise ValueError("img_size is too small for the configured first-stage patch embedding")
        for _ in range(1, self.num_stages):
            stage_size = _conv_output_size(stage_size, 3, 2, 1)
            if stage_size <= 0:
                raise ValueError(
                    "img_size does not provide enough spatial resolution for the configured number of PVT_V2 stages"
                )


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_pvt_config(
    *,
    img_size: int,
    num_classes: int,
    patch_size: int,
    embed_dims: tuple[int, ...] | list[int],
    num_heads: tuple[int, ...] | list[int],
    mlp_ratios: tuple[float, ...] | list[float],
    qkv_bias: bool,
    depths: tuple[int, ...] | list[int],
    sr_ratios: tuple[int, ...] | list[int],
    **kwargs,
) -> PVTConfig:
    params = {
        "img_size": img_size,
        "num_classes": num_classes,
        "patch_size": patch_size,
        "embed_dims": embed_dims,
        "num_heads": num_heads,
        "mlp_ratios": mlp_ratios,
        "qkv_bias": qkv_bias,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "depths": depths,
        "sr_ratios": sr_ratios,
    }
    params.update(kwargs)
    return PVTConfig(**params)


def _build_pvt_v2_config(
    *,
    img_size: int,
    num_classes: int,
    in_channels: int,
    patch_size: int,
    embed_dims: tuple[int, ...] | list[int],
    num_heads: tuple[int, ...] | list[int],
    mlp_ratios: tuple[int, ...] | list[int],
    qkv_bias: bool,
    depths: tuple[int, ...] | list[int],
    sr_ratios: tuple[int, ...] | list[int],
    num_stages: int,
    linear: bool = False,
    **kwargs,
) -> PVTV2Config:
    params = {
        "img_size": img_size,
        "patch_size": patch_size,
        "in_channels": in_channels,
        "num_classes": num_classes,
        "embed_dims": embed_dims,
        "num_heads": num_heads,
        "mlp_ratios": mlp_ratios,
        "qkv_bias": qkv_bias,
        "norm_layer": partial(nn.LayerNorm, eps=1e-6),
        "depths": depths,
        "sr_ratios": sr_ratios,
        "num_stages": num_stages,
        "linear": linear,
    }
    params.update(kwargs)
    return PVTV2Config(**params)


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class _SRAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} should be divided by num_heads {num_heads}.")

        self.dim = dim
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .transpose((0, 2, 1, 3))
        )

        if self.sr_ratio > 1:
            x_ = x.transpose((0, 2, 1)).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose((0, 2, 1))
            x_ = self.norm(x_)

            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .transpose((2, 0, 3, 1, 4))
            )
        else:
            kv = (
                self.kv(x)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .transpose((2, 0, 3, 1, 4))
            )

        k, v = kv[0], kv[1]
        attn = (q @ k.mT) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        sr_ratio: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _SRAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x += self.drop_path(self.attn(self.norm1(x), H, W))
        x += self.drop_path(self.mlp(self.norm2(x)))

        return x


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        if img_size[0] % patch_size[0] and img_size[1] % patch_size[1]:
            raise ValueError(
                f"img_size {img_size} should be divided by patch_size {patch_size}."
            )

        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        x = self.proj(x)

        x = x.reshape(*x.shape[:2], -1).swapaxes(1, 2)
        x = self.norm(x)

        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class PVT(nn.Module):
    def __init__(self, config: PVTConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.depths = config.depths

        stage1_size = _conv_output_size(
            config.img_size,
            config.patch_size,
            config.patch_size,
        )
        stage2_size = _conv_output_size(stage1_size, 2, 2)
        stage3_size = _conv_output_size(stage2_size, 2, 2)

        self.patch_emb1 = _PatchEmbed(
            config.img_size,
            config.patch_size,
            config.in_channels,
            config.embed_dims[0],
        )
        self.patch_emb2 = _PatchEmbed(stage1_size, 2, config.embed_dims[0], config.embed_dims[1])
        self.patch_emb3 = _PatchEmbed(stage2_size, 2, config.embed_dims[1], config.embed_dims[2])
        self.patch_emb4 = _PatchEmbed(stage3_size, 2, config.embed_dims[2], config.embed_dims[3])

        self.pos_emb1 = nn.Parameter(
            lucid.zeros(1, self.patch_emb1.num_patches, config.embed_dims[0])
        )
        self.pos_drop1 = nn.Dropout(config.drop_rate)

        self.pos_emb2 = nn.Parameter(
            lucid.zeros(1, self.patch_emb2.num_patches, config.embed_dims[1])
        )
        self.pos_drop2 = nn.Dropout(config.drop_rate)

        self.pos_emb3 = nn.Parameter(
            lucid.zeros(1, self.patch_emb3.num_patches, config.embed_dims[2])
        )
        self.pos_drop3 = nn.Dropout(config.drop_rate)

        self.pos_emb4 = nn.Parameter(
            lucid.zeros(1, self.patch_emb4.num_patches + 1, config.embed_dims[3])
        )
        self.pos_drop4 = nn.Dropout(config.drop_rate)

        dpr = [
            x.item()
            for x in lucid.linspace(0, config.drop_path_rate, sum(config.depths))
        ]
        cur = 0
        for i in range(4):
            blocks_ = [
                _Block(
                    config.embed_dims[i],
                    config.num_heads[i],
                    config.mlp_ratios[i],
                    config.qkv_bias,
                    config.qk_scale,
                    config.drop_rate,
                    config.attn_drop_rate,
                    dpr[cur + j],
                    norm_layer=config.norm_layer,
                    sr_ratio=config.sr_ratios[i],
                )
                for j in range(config.depths[i])
            ]
            self.add_module(f"block{i + 1}", nn.ModuleList(blocks_))
            cur += config.depths[i]

        self.norm = config.norm_layer(config.embed_dims[3])
        self.head = (
            nn.Linear(config.embed_dims[3], config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )

        self.cls_token = nn.Parameter(lucid.zeros(1, 1, config.embed_dims[3]))
        nn.init.normal(self.cls_token, std=0.02)

        for pos_emb in [self.pos_emb1, self.pos_emb2, self.pos_emb3, self.pos_emb4]:
            nn.init.normal(pos_emb, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant(m.bias, 0.0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant(m.bias, 0.0)
            nn.init.constant(m.weight, 1.0)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        for i in range(4):
            x, (H, W) = getattr(self, f"patch_emb{i + 1}")(x)
            if i == 3:
                cls_tokens = self.cls_token.repeat(B, axis=0)
                x = lucid.concatenate([cls_tokens, x], axis=1)

            x += getattr(self, f"pos_emb{i + 1}")
            x = getattr(self, f"pos_drop{i + 1}")(x)

            for block in getattr(self, f"block{i + 1}"):
                x = block(x, H, W)

            x = (
                x.reshape(B, H, W, -1).transpose((0, 3, 1, 2))
                if i < 3
                else self.norm(x)
            )

        x = self.head(x[:, 0])
        return x


class _DWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, groups=dim
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, _, C = x.shape
        x = x.swapaxes(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(start_axis=2).swapaxes(1, 2)

        return x


class _ConvMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
        linear: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = _DWConv(hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU()

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant(m.bias, 0)
            nn.init.constant(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.zero()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)

        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x


class _LSRAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        sr_ratio: int = 1,
        linear: bool = False,
    ) -> None:
        super().__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .transpose((0, 2, 1, 3))
        )

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.transpose((0, 2, 1)).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).transpose((0, 2, 1))
                x_ = self.norm(x_)
                kv = (
                    self.kv(x_)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .transpose((2, 0, 3, 1, 4))
                )
            else:
                kv = (
                    self.kv(x)
                    .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                    .transpose((2, 0, 3, 1, 4))
                )
        else:
            x_ = x.transpose((0, 2, 1)).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).transpose((0, 2, 1))
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = (
                self.kv(x_)
                .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
                .transpose((2, 0, 3, 1, 4))
            )

        k, v = kv[0], kv[1]
        attn = (q @ k.mT) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _Block_V2(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        sr_ratio: int = 1,
        linear: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _LSRAttention(
            dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, sr_ratio, linear
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _ConvMLP(
            dim, mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x += self.drop_path(self.attn(self.norm1(x), H, W))
        x += self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class _OverlapPatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 7,
        stride: int = 4,
        in_channels: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride."

        self.img_size = img_size
        self.patch_size = patch_size

        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2),
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:
        x = self.proj(x)
        H, W = x.shape[2:]
        x = x.flatten(start_axis=2).swapaxes(1, 2)
        x = self.norm(x)

        return x, H, W


class PVT_V2(nn.Module):
    def __init__(self, config: PVTV2Config) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.depths = config.depths
        self.num_stages = config.num_stages

        dpr = [
            x.item()
            for x in lucid.linspace(0, config.drop_path_rate, sum(config.depths))
        ]
        cur = 0
        stage_img_size = config.img_size

        for i in range(config.num_stages):
            stage_patch_size = config.patch_size if i == 0 else 3
            stage_stride = 4 if i == 0 else 2
            patch_embed = _OverlapPatchEmbed(
                img_size=stage_img_size,
                patch_size=stage_patch_size,
                stride=stage_stride,
                in_channels=config.in_channels if i == 0 else config.embed_dims[i - 1],
                embed_dim=config.embed_dims[i],
            )
            block = [
                _Block_V2(
                    dim=config.embed_dims[i],
                    num_heads=config.num_heads[i],
                    mlp_ratio=config.mlp_ratios[i],
                    qkv_bias=config.qkv_bias,
                    qk_scale=config.qk_scale,
                    drop=config.drop_rate,
                    attn_drop=config.attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=config.norm_layer,
                    sr_ratio=config.sr_ratios[i],
                    linear=config.linear,
                )
                for j in range(config.depths[i])
            ]
            block = nn.ModuleList(block)

            norm = config.norm_layer(config.embed_dims[i])
            cur += config.depths[i]
            stage_img_size = _conv_output_size(
                stage_img_size,
                stage_patch_size,
                stage_stride,
                stage_patch_size // 2,
            )

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.head = (
            nn.Linear(config.embed_dims[config.num_stages - 1], config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant(m.bias, 0)
            nn.init.constant(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            nn.init.normal(m.weight, 0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.zero()

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")

            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)

            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape(B, H, W, -1).transpose((0, 3, 1, 2))

        x = x.mean(axis=1)
        x = self.head(x)

        return x


@register_model
def pvt_tiny(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, or sr_ratios",
    )
    config = _build_pvt_config(
        img_size=img_size,
        num_classes=num_classes,
        patch_size=4,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        **kwargs,
    )
    return PVT(config)


@register_model
def pvt_small(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, or sr_ratios",
    )
    config = _build_pvt_config(
        img_size=img_size,
        num_classes=num_classes,
        patch_size=4,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        **kwargs,
    )
    return PVT(config)


@register_model
def pvt_medium(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, or sr_ratios",
    )
    config = _build_pvt_config(
        img_size=img_size,
        num_classes=num_classes,
        patch_size=4,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 4, 18, 3),
        sr_ratios=(8, 4, 2, 1),
        **kwargs,
    )
    return PVT(config)


@register_model
def pvt_large(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, or sr_ratios",
    )
    config = _build_pvt_config(
        img_size=img_size,
        num_classes=num_classes,
        patch_size=4,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 4, 27, 3),
        sr_ratios=(8, 4, 2, 1),
        **kwargs,
    )
    return PVT(config)


@register_model
def pvt_huge(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "drop_path_rate"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or drop_path_rate",
    )
    config = _build_pvt_config(
        img_size=img_size,
        num_classes=num_classes,
        patch_size=4,
        embed_dims=(128, 256, 512, 768),
        num_heads=(2, 4, 8, 12),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 10, 60, 3),
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.02,
        **kwargs,
    )
    return PVT(config)


@register_model
def pvt_v2_b0(
    img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PVT_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "num_stages"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or num_stages",
    )
    config = _build_pvt_v2_config(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=7,
        embed_dims=(32, 64, 160, 256),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        num_stages=4,
        **kwargs,
    )
    return PVT_V2(config)


@register_model
def pvt_v2_b1(
    img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PVT_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "num_stages"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or num_stages",
    )
    config = _build_pvt_v2_config(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=7,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(2, 2, 2, 2),
        sr_ratios=(8, 4, 2, 1),
        num_stages=4,
        **kwargs,
    )
    return PVT_V2(config)


@register_model
def pvt_v2_b2(
    img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PVT_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "num_stages"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or num_stages",
    )
    config = _build_pvt_v2_config(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=7,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        num_stages=4,
        **kwargs,
    )
    return PVT_V2(config)


@register_model
def pvt_v2_b2_li(
    img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PVT_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "num_stages", "linear"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, num_stages, or linear",
    )
    config = _build_pvt_v2_config(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=7,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 4, 6, 3),
        sr_ratios=(8, 4, 2, 1),
        num_stages=4,
        linear=True,
        **kwargs,
    )
    return PVT_V2(config)


@register_model
def pvt_v2_b3(
    img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PVT_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "num_stages"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or num_stages",
    )
    config = _build_pvt_v2_config(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=7,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 4, 18, 3),
        sr_ratios=(8, 4, 2, 1),
        num_stages=4,
        **kwargs,
    )
    return PVT_V2(config)


@register_model
def pvt_v2_b4(
    img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PVT_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "num_stages"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or num_stages",
    )
    config = _build_pvt_v2_config(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=7,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 8, 27, 3),
        sr_ratios=(8, 4, 2, 1),
        num_stages=4,
        **kwargs,
    )
    return PVT_V2(config)


@register_model
def pvt_v2_b5(
    img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs
) -> PVT_V2:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"patch_size", "embed_dims", "num_heads", "mlp_ratios", "qkv_bias", "norm_layer", "depths", "sr_ratios", "num_stages"},
        "factory variants do not allow overriding preset patch_size, embed_dims, num_heads, mlp_ratios, qkv_bias, norm_layer, depths, sr_ratios, or num_stages",
    )
    config = _build_pvt_v2_config(
        img_size=img_size,
        num_classes=num_classes,
        in_channels=in_channels,
        patch_size=7,
        embed_dims=(64, 128, 320, 512),
        num_heads=(1, 2, 5, 8),
        mlp_ratios=(8, 8, 4, 4),
        qkv_bias=True,
        depths=(3, 6, 40, 3),
        sr_ratios=(8, 4, 2, 1),
        num_stages=4,
        **kwargs,
    )
    return PVT_V2(config)
