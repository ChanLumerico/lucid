from typing import Callable
from functools import partial
import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["EfficientFormer"]


def _to_2_tuple(val: int | float) -> tuple:
    return (val, val)


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
        pos = lucid.stack([y, x]).flatten(axis=1)
        rel_pos = lucid.abs(pos[..., :, None] - pos[..., None, :])
        rel_pos = (rel_pos[0] * resolution[1]) + rel_pos[1]

        self.attention_biases = nn.Parameter(
            lucid.zeros(num_heads, resolution[0] * resolution[1])
        )
        self.attention_bias_idxs: nn.Buffer
        self.register_buffer("attention_bias_idxs", rel_pos)

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


class _Stem4(nn.Sequential):
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
        x = x.flatten(axis=2).swapaxes(1, 2)
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
    NotImplemented


class _MetaBlock2d(nn.Module):
    NotImplemented


class _EfficientFormerStage(nn.Module):
    NotImplemented


class EfficientFormer(nn.Module):
    NotImplemented
