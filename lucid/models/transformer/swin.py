from typing import Type

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


def _to_2tuple(val: int | float) -> tuple[int | float, ...]:
    return (val, val)


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
    def __init__(
        self,
        dim: int,
        input_res: tuple[int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: int = 4,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_res = input_res
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_res) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_res)

        if not (0 <= self.shift_size < self.window_size):
            raise ValueError("shift_size must be in [0, window_size).")

        self.norm1 = norm_layer(dim)
        self.attn = _WindowAttention(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W = self.input_res
            img_mask = lucid.zeros(1, H, W, 1)
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(-1, self.window_size * self.window_size)

            attn_mask = mask_windows.unsqueeze(axis=1) - mask_windows.unsqueeze(axis=2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
                attn_mask == 0, 0.0
            )
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: Tensor) -> Tensor:
        H, W = self.input_res
        B, L, C = x.shape
        assert L == H * W, "wrong input feature size."

        shortcut = x
        x = self.norm1(x)
        x = x.reshape(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = lucid.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2)
            )
            x_windows = window_partition(shifted_x, self.window_size)
        else:
            shifted_x = x
            x_windows = window_partition(shifted_x, self.window_size)

        x_windows = x_windows.reshape(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.reshape(-1, self.window_size, self.window_size, C)

        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = lucid.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), axis=(1, 2)
            )
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)
            x = shifted_x

        x = x.reshape(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x += self.drop_path(self.mlp(self.norm2(x)))

        return x


class _PatchMerging(nn.Module):
    NotImplemented


class _BaseLayer(nn.Module):
    NotImplemented


class _PatchEmbed(nn.Module):
    NotImplemented


class SwinTransformer(nn.Module):
    NotImplemented
