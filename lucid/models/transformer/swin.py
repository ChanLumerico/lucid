from typing import Type

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


def window_partition(x: Tensor, window_size: int) -> Tensor:
    B, H, W, C = x.shape
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)

    windows = x.swapaxes(2, 3).reshape(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: Tensor, window_size: int, H: int, W: int) -> Tensor:
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.swapaxes(2, 3).reshape(B, H, W, -1)
    return x


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: Type[nn.Module] = nn.GELU,
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


class _WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.relative_position_bias_table = nn.Parameter(
            lucid.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        nn.init.normal(self.relative_position_bias_table, std=0.02)

        coords_h = lucid.arange(self.window_size[0])
        coords_w = lucid.arange(self.window_size[1])

        coords = lucid.stack(lucid.meshgrid(coords_h, coords_w))
        coords_flatten = coords.reshape(coords.shape[0], -1)

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.transpose((1, 2, 0))

        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_pos_index = relative_coords.sum(axis=-1)
        self.register_buffer("relative_pos_index", relative_pos_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q *= self.scale
        attn = q @ k.mT

        relative_pos_bias = self.relative_position_bias_table[
            self.relative_pos_index.flatten().astype(int)
        ].reshape(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_pos_bias = relative_pos_bias.transpose((2, 0, 1))
        attn += relative_pos_bias.unsqueeze(axis=0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N)
            attn += mask.unsqueeze(axis=1).unsqueeze(axis=0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _SwinTransformerBlock(nn.Module):
    NotImplemented
