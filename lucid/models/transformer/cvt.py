from typing import Literal, Type

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


def _to_2tuple(val: int | float) -> tuple[int | float, ...]:
    return (val, val)


class _QuickGELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * F.sigmoid(1.702 * x)


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


class _ConvAttention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        method: Literal["dw_bn", "avg", "lin"] = "dw_bn",
        kernel_size: int = 3,
        stride_kv: int = 1,
        stride_q: int = 1,
        padding_kv: int = 1,
        padding_q: int = 1,
        with_cls_token: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads

        self.scale = dim_out**-0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in,
            kernel_size,
            padding_q,
            stride_q,
            method="lin" if method == "avg" else method,
        )
        self.conv_proj_k = self._build_projection(
            dim_in, kernel_size, padding_kv, stride_kv, method
        )
        self.conv_proj_v = self._build_projection(
            dim_in, kernel_size, padding_kv, stride_kv, method
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(
        self,
        dim_in: int,
        kernel_size: int,
        padding: int,
        stride: int,
        method: str,
    ) -> nn.Sequential | None:
        match method:
            case "dw_bn":
                proj = nn.Sequential(
                    nn.Conv2d(
                        dim_in,
                        dim_in,
                        kernel_size,
                        stride,
                        padding,
                        groups=dim_in,
                        bias=False,
                    ),
                    nn.BatchNorm2d(dim_in),
                    nn.Rearrange("b c h w -> b (h w) c"),
                )
            case "avg":
                proj = nn.Sequential(
                    nn.AvgPool2d(kernel_size, stride, padding),
                    nn.Rearrange("b c h w -> b (h w) c"),
                )
            case "lin":
                proj = None
            case _:
                raise ValueError(f"unknown method ({method}).")

        return proj

    def forward_conv(self, x: Tensor, h: int, w: int) -> tuple[Tensor, Tensor, Tensor]:
        if self.with_cls_token:
            cls_token, x = lucid.split(x, [1, h * w], axis=1)

        x = lucid.einops.rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        qkv = []
        for conv_proj in [self.conv_proj_q, self.conv_proj_k, self.conv_proj_v]:
            if conv_proj is not None:
                _proj = conv_proj(x)
            else:
                _proj = lucid.einops.rearrange(x, "b c h w -> b (h w) c")
            qkv.append(_proj)

        q, k, v = tuple(qkv)

        if self.with_cls_token:
            q = lucid.concatenate([cls_token, q], axis=1)
            k = lucid.concatenate([cls_token, k], axis=1)
            v = lucid.concatenate([cls_token, v], axis=1)

        return q, k, v

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        if not (
            self.conv_proj_q is None
            and self.conv_proj_k is None
            and self.conv_proj_k is None
        ):
            q, k, v = self.forward_conv(x, h, w)

        q = self.proj_q(q)
        k = self.proj_k(k)
        v = self.proj_v(v)

        q = lucid.einops.rearrange(
            q, "b t (h d) -> b h t d", h=self.num_heads, d=q.shape[-1] // self.num_heads
        )
        k = lucid.einops.rearrange(
            k, "b t (h d) -> b h t d", h=self.num_heads, d=k.shape[-1] // self.num_heads
        )
        v = lucid.einops.rearrange(
            v, "b t (h d) -> b h t d", h=self.num_heads, d=v.shape[-1] // self.num_heads
        )

        attn_score = (q @ k.mT) * self.scale
        attn = F.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)

        x = attn @ v
        x = lucid.einops.rearrange(x, "b h t d -> b t (h d)")

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class _ConvTransformerBlock(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__()
        self.with_cls_token = kwargs["with_cls_token"]

        self.norm1 = norm_layer(dim_in)
        self.attn = _ConvAttention(
            dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs
        )

        self.drop_path = nn.DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = _MLP(dim_out, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        res = x
        x = self.norm1(x)
        attn = self.attn(x, h, w)

        x = res + self.drop_path(attn)
        x += self.drop_path(self.mlp(self.norm2(x)))

        return x


class _ConvEmbed(nn.Module):
    def __init__(
        self,
        patch_size: int = 7,
        in_channels: int = 3,
        embed_dim: int = 64,
        stride: int = 4,
        padding: int = 2,
        norm_layer: Type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        patch_size = _to_2tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        _, _, H, W = x.shape
        x = lucid.einops.rearrange(x, "b c h w -> b (h w) c")

        if self.norm is not None:
            x = self.norm(x)

        x = lucid.einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)
        return x


class _VisionTransformer(nn.Module): ...


class CvT(nn.Module): ...
