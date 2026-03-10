from functools import partial
from typing import Callable, Literal, Type
from dataclasses import dataclass, field

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

__all__ = ["CvT", "CvTConfig", "cvt_13", "cvt_21", "cvt_w24"]


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


class _VisionTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int = 16,
        patch_stride: int = 16,
        patch_padding: int = 0,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.rearrange = None

        self.patch_embed = _ConvEmbed(
            patch_size,
            in_channels,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:
            self.cls_token = nn.Parameter(lucid.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in lucid.linspace(0, drop_path_rate, depth)]

        blocks = []
        for j in range(depth):
            blocks.append(
                _ConvTransformerBlock(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            nn.init.normal(self.cls_token, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        B, _, H, W = x.shape

        x = lucid.einops.rearrange(x, "b c h w -> b (h w) c")

        cls_tokens = None
        if self.cls_token is not None:
            cls_tokens = self.cls_token.repeat(B, axis=0)
            x = lucid.concatenate([cls_tokens, x], axis=1)

        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = lucid.split(x, [1, H * W], axis=1)

        x = lucid.einops.rearrange(x, "b (h w) c -> b c h w", h=H, w=W)

        return x, cls_tokens


@dataclass
class CvTConfig:
    num_stages: int
    patch_size: tuple[int, ...] | list[int]
    patch_stride: tuple[int, ...] | list[int]
    patch_padding: tuple[int, ...] | list[int]
    dim_embed: tuple[int, ...] | list[int]
    num_heads: tuple[int, ...] | list[int]
    depth: tuple[int, ...] | list[int]
    in_channels: int = 3
    num_classes: int = 1000
    act_layer: Callable[..., nn.Module] = nn.GELU
    norm_layer: Callable[..., nn.Module] = nn.LayerNorm

    mlp_ratio: tuple[float, ...] | list[float] = field(
        default_factory=lambda: [4.0, 4.0, 4.0]
    )
    attn_drop_rate: tuple[float, ...] | list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    drop_rate: tuple[float, ...] | list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.0]
    )
    drop_path_rate: tuple[float, ...] | list[float] = field(
        default_factory=lambda: [0.0, 0.0, 0.1]
    )

    qkv_bias: tuple[bool, ...] | list[bool] = field(
        default_factory=lambda: [True, True, True]
    )
    cls_token: tuple[bool, ...] | list[bool] = field(
        default_factory=lambda: [False, False, True]
    )
    pos_embed: tuple[bool, ...] | list[bool] = field(
        default_factory=lambda: [False, False, False]
    )

    qkv_proj_method: tuple[str, ...] | list[str] = field(
        default_factory=lambda: ["dw_bn", "dw_bn", "dw_bn"]
    )

    kernel_qkv: tuple[int, ...] | list[int] = field(default_factory=lambda: [3, 3, 3])
    padding_kv: tuple[int, ...] | list[int] = field(default_factory=lambda: [1, 1, 1])
    stride_kv: tuple[int, ...] | list[int] = field(default_factory=lambda: [2, 2, 2])
    padding_q: tuple[int, ...] | list[int] = field(default_factory=lambda: [1, 1, 1])
    stride_q: tuple[int, ...] | list[int] = field(default_factory=lambda: [1, 1, 1])

    def __post_init__(self) -> None:
        stage_sequences = (
            "patch_size",
            "patch_stride",
            "patch_padding",
            "dim_embed",
            "num_heads",
            "depth",
            "mlp_ratio",
            "attn_drop_rate",
            "drop_rate",
            "drop_path_rate",
            "qkv_bias",
            "cls_token",
            "pos_embed",
            "qkv_proj_method",
            "kernel_qkv",
            "padding_kv",
            "stride_kv",
            "padding_q",
            "stride_q",
        )
        for field_name in stage_sequences:
            setattr(self, field_name, tuple(getattr(self, field_name)))

        if self.num_stages <= 0:
            raise ValueError("num_stages must be greater than 0")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.num_classes < 0:
            raise ValueError("num_classes must be greater than or equal to 0")

        for field_name in stage_sequences:
            if len(getattr(self, field_name)) != self.num_stages:
                raise ValueError(
                    f"{field_name} must contain exactly num_stages values"
                )

        positive_int_fields = (
            "patch_size",
            "patch_stride",
            "dim_embed",
            "num_heads",
            "depth",
            "kernel_qkv",
            "stride_kv",
            "stride_q",
        )
        for field_name in positive_int_fields:
            values = getattr(self, field_name)
            if any(value <= 0 for value in values):
                raise ValueError(f"{field_name} must contain positive integers")
        for field_name in ("patch_padding", "padding_kv", "padding_q"):
            values = getattr(self, field_name)
            if any(value < 0 for value in values):
                raise ValueError(f"{field_name} must contain non-negative integers")

        if any(ratio <= 0 for ratio in self.mlp_ratio):
            raise ValueError("mlp_ratio must contain positive values")
        for field_name in ("attn_drop_rate", "drop_rate", "drop_path_rate"):
            values = getattr(self, field_name)
            if any(value < 0 or value >= 1 for value in values):
                raise ValueError(f"{field_name} values must be in the range [0, 1)")

        if any(
            method not in {"dw_bn", "avg", "lin"} for method in self.qkv_proj_method
        ):
            raise ValueError("qkv_proj_method values must be dw_bn, avg, or lin")
        for dim, heads in zip(self.dim_embed, self.num_heads):
            if dim % heads != 0:
                raise ValueError("dim_embed values must be divisible by num_heads")


class CvT(nn.Module):
    def __init__(self, config: CvTConfig) -> None:
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.num_stages = config.num_stages
        in_channels = config.in_channels

        for i in range(self.num_stages):
            kwargs = dict(
                patch_size=config.patch_size[i],
                patch_stride=config.patch_stride[i],
                patch_padding=config.patch_padding[i],
                embed_dim=config.dim_embed[i],
                depth=config.depth[i],
                num_heads=config.num_heads[i],
                mlp_ratio=config.mlp_ratio[i],
                qkv_bias=config.qkv_bias[i],
                drop_rate=config.drop_rate[i],
                attn_drop_rate=config.attn_drop_rate[i],
                drop_path_rate=config.drop_path_rate[i],
                with_cls_token=config.cls_token[i],
                method=config.qkv_proj_method[i],
                kernel_size=config.kernel_qkv[i],
                padding_q=config.padding_q[i],
                padding_kv=config.padding_kv[i],
                stride_kv=config.stride_kv[i],
                stride_q=config.stride_q[i],
            )

            stage = _VisionTransformer(
                in_channels=in_channels,
                act_layer=config.act_layer,
                norm_layer=config.norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_channels = config.dim_embed[i]

        dim_embed = config.dim_embed[-1]
        self.norm = config.norm_layer(dim_embed)
        self.cls_token = config.cls_token[-1]

        self.head = (
            nn.Linear(dim_embed, config.num_classes)
            if config.num_classes > 0
            else nn.Identity()
        )
        if isinstance(self.head, nn.Linear):
            nn.init.normal(self.head.weight, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f"stage{i}")(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = lucid.squeeze(x, axis=1)
        else:
            x = lucid.einops.rearrange(x, "b c h w -> b (h w) c")
            x = self.norm(x)
            x = lucid.mean(x, axis=1)

        x = self.head(x)
        return x


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_cvt_config(num_classes: int, **kwargs) -> CvTConfig:
    params = {
        "in_channels": 3,
        "num_classes": num_classes,
        "act_layer": _QuickGELU,
        "norm_layer": partial(nn.LayerNorm, eps=1e-5),
    }
    params.update(kwargs)
    return CvTConfig(**params)


@register_model
def cvt_13(num_classes: int = 1000, **kwargs) -> CvT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"num_stages", "patch_size", "patch_stride", "patch_padding", "dim_embed", "num_heads", "depth"},
        "factory variants do not allow overriding preset num_stages, patch_size, patch_stride, patch_padding, dim_embed, num_heads, or depth",
    )
    config = _build_cvt_config(
        num_classes,
        num_stages=3,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        dim_embed=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 2, 10],
        **kwargs,
    )
    return CvT(config)


@register_model
def cvt_21(num_classes: int = 1000, **kwargs) -> CvT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"num_stages", "patch_size", "patch_stride", "patch_padding", "dim_embed", "num_heads", "depth"},
        "factory variants do not allow overriding preset num_stages, patch_size, patch_stride, patch_padding, dim_embed, num_heads, or depth",
    )
    config = _build_cvt_config(
        num_classes,
        num_stages=3,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        dim_embed=[64, 192, 384],
        num_heads=[1, 3, 6],
        depth=[1, 4, 16],
        **kwargs,
    )
    return CvT(config)


@register_model
def cvt_w24(num_classes: int = 1000, **kwargs) -> CvT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"num_stages", "patch_size", "patch_stride", "patch_padding", "dim_embed", "num_heads", "depth"},
        "factory variants do not allow overriding preset num_stages, patch_size, patch_stride, patch_padding, dim_embed, num_heads, or depth",
    )
    config = _build_cvt_config(
        num_classes,
        num_stages=3,
        patch_size=[7, 3, 3],
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        dim_embed=[192, 768, 1024],
        num_heads=[3, 12, 16],
        depth=[2, 2, 20],
        drop_path_rate=[0.0, 0.0, 0.3],
        **kwargs,
    )
    return CvT(config)
