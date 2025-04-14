from functools import partial
import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

from .pvt import _MLP


def _make_divisible(v: int, divisor: int = 8, min_value: int | None = None) -> int:
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor

    return new_v


class _SE(nn.Module):
    def __init__(self, dim: int, hidden_ratio: int | None = None) -> None:
        super().__init__()
        hidden_ratio = hidden_ratio or 1
        self.dim = dim

        hidden_dim = int(dim * hidden_ratio)
        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),
        )

    def forward(self, x: Tensor) -> Tensor:
        a = x.mean(axis=1, keepdims=True)
        a = self.fc(a)
        x *= a
        return x


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        head_dim = hidden_dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim**-0.5

        self.qk = nn.Linear(dim, hidden_dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N = x.shape[:2]
        qk = (
            self.qk(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .transpose((2, 0, 3, 1, 4))
        )
        q, k = qk[0], qk[1]
        v = self.v(x).reshape(B, N, self.num_heads, -1).transpose((0, 2, 1, 3))

        attn = (q @ k.mT) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _TNTBlock(nn.Module):
    def __init__(
        self,
        outer_dim: int,
        inner_dim: int,
        outer_num_heads: int,
        inner_num_heads: int,
        num_words: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        se: int = 0,
    ) -> None:
        super().__init__()
        self.has_inner = inner_dim > 0
        if self.has_inner:
            self.inner_norm1 = norm_layer(inner_dim)
            self.inner_attn = _Attention(
                inner_dim,
                inner_dim,
                num_heads=inner_num_heads,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.inner_norm2 = norm_layer(inner_dim)
            ...

            # TODO: Continue from here
