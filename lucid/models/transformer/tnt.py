from functools import partial
from typing import Any
import math

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


def exists(val: Any) -> bool:
    return val is not None


def default(val: Any, d: Any) -> Any:
    return val if exists(val) else d


def divisible_by(val: int, divisor: int) -> bool:
    return (val % divisor) == 0


def unfold_output_size(
    image_size: int, kernel_size: int, stride: int, padding: int
) -> int:
    return int((image_size - kernel_size + (2 * padding) / stride) + 1)


class _PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        return self.fn(self.norm(x), **kwargs)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, multiply: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * multiply),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * multiply, dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _Attention(nn.Module):
    def __init__(
        self, *, dim: int, num_heads: int = 8, dim_head: int = 64, dropout: float = 0.0
    ) -> None:
        super().__init__()
        inner_dim = num_heads * dim_head
        self.num_heads = num_heads
        self.scale = dim_head**-0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _, h = *x.shape, self.num_heads
        q, k, v = self.to_qkv(x).chunk(3, axis=-1)
        q, k, v = map(
            lambda t: lucid.einops.rearrange(
                t, "b n (h d) -> (b h) n d", h=h, d=t.shape[2] // h
            ),
            (q, k, v),
        )
        sim = (q @ k.mT) * self.scale
        attn = F.softmax(sim, axis=-1)

        out = attn @ v
        out = lucid.einops.rearrange(out, "(b h) n d -> b n (h d)", h=h, b=b)

        return self.to_out(out)
