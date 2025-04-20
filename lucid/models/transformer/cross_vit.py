from typing import Any
from functools import partial

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


def _to_2tuple(val: Any) -> tuple[Any, Any]:
    return (val, val)


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        multi_conv: bool = False,
    ) -> None:
        super().__init__()
        img_size = _to_2tuple(img_size)
        patch_size = _to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if multi_conv:
            if patch_size[0] == 12:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim // 4, 7, 4, 3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 3, 0),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, 3, 1, 1),
                )

            elif patch_size[0] == 16:
                self.proj = nn.Sequential(
                    nn.Conv2d(in_channels, embed_dim // 4, 7, 4, 3),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 4, embed_dim // 2, 3, 2, 1),
                    nn.ReLU(),
                    nn.Conv2d(embed_dim // 2, embed_dim, 3, 2, 1),
                )
        else:
            self.proj = nn.Conv2d(
                in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
            )

    def forward(self, x: Tensor) -> Tensor:
        H, W = x.shape[2:]
        if H != self.img_size[0] or W != self.img_size[1]:
            raise ValueError(
                f"Input image size {(H, W)} does not match with {self.img_size}."
            )

        x = self.proj(x).flatten(axis=2).swapaxes(1, 2)
        return x


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
        use_conv: bool = False,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        bias = _to_2tuple(bias)
        drop_probs = _to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])

        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        x = self.drop2(self.fc2(self.norm(x)))

        return x


class _LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(init_value * lucid.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads."

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .transpose((2, 0, 3, 1, 4))
        )
        q, k, v = qkv.chunk(3)
        q, k = self.q_norm(q), self.k_norm(k)

        attn = q @ k.mT * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = x.reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_bias: bool = True,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_value: float | None = None,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        mlp_layer: type[nn.Module] = _MLP,
    ) -> None:
        super().__init__()

        # TODO: Finish this class and move on to `_CrossAttentionBlock`


class _CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        q = (
            self.wq(x[:, 0:1, ...])
            .reshape(B, 1, self.num_heads, C // self.num_heads)
            .swapaxes(1, 2)
        )
        k = self.wk(x).reshape(B, N, self.num_heads, C // self.num_heads).swapaxes(1, 2)
        v = self.wv(x).reshape(B, N, self.num_heads, C // self.num_heads).swapaxes(1, 2)

        attn = (q @ k.mT) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _CrossAttentionBlock(nn.Module):
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
        act_layer: type[nn.Module] = nn.ReLU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        has_mlp: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = _CrossAttention(
            dim, num_heads, qkv_bias, qkv_bias, attn_drop, proj_drop=drop
        )
        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.has_mlp = has_mlp
        if has_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = ...
