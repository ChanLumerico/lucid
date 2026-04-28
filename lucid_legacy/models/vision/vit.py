from dataclasses import dataclass

import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor

__all__ = [
    "ViT",
    "ViTConfig",
    "vit_tiny",
    "vit_small",
    "vit_base",
    "vit_large",
    "vit_huge",
]


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    embedding_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout_rate: float = 0.1

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than 0")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be greater than 0")
        if self.image_size % self.patch_size != 0:
            raise ValueError("image_size must be divisible by the patch_size.")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be greater than 0")
        if self.num_classes <= 0:
            raise ValueError("num_classes must be greater than 0")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than 0")
        if self.depth <= 0:
            raise ValueError("depth must be greater than 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be greater than 0")
        if self.embedding_dim % self.num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        if self.mlp_dim <= 0:
            raise ValueError("mlp_dim must be greater than 0")
        if self.dropout_rate < 0 or self.dropout_rate >= 1:
            raise ValueError("dropout_rate must be in the range [0, 1)")


class ViT(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config

        self.embedding_dim = config.embedding_dim
        self.num_patches = (config.image_size // config.patch_size) ** 2
        self.patch_emb = nn.Conv2d(
            config.in_channels,
            config.embedding_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )

        self.cls_token = nn.Parameter(lucid.zeros(1, 1, config.embedding_dim))
        self.pos_emb = nn.Parameter(
            lucid.zeros(1, 1 + self.num_patches, config.embedding_dim)
        )

        nn.init.normal(self.cls_token, std=0.02)
        nn.init.normal(self.pos_emb, std=0.02)

        self.dropout = nn.Dropout(config.dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            num_heads=config.num_heads,
            dim_feedforward=config.mlp_dim,
            dropout=config.dropout_rate,
            activation=F.gelu,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.depth
        )

        self.norm = nn.LayerNorm(config.embedding_dim)
        self.head = nn.Linear(config.embedding_dim, config.num_classes)

    def forward(self, x: Tensor) -> Tensor:
        N = x.shape[0]
        x = self.patch_emb(x)
        x = lucid.einops.rearrange(x, "n d h w -> n (h w) d")

        cls_tokens = self.cls_token.repeat(N, axis=0)
        x = lucid.concatenate([cls_tokens, x], axis=1)

        x += self.pos_emb
        x = self.dropout(x)

        x = self.transformer_encoder(x)
        x = self.norm(x)

        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits


def _raise_for_locked_factory_kwargs(
    kwargs: dict[str, object],
    locked_fields: set[str],
    message: str,
) -> None:
    if locked_fields & kwargs.keys():
        raise TypeError(message)


def _build_vit_config(
    *,
    image_size: int,
    patch_size: int,
    num_classes: int,
    embedding_dim: int,
    depth: int,
    num_heads: int,
    mlp_dim: int,
    **kwargs,
) -> ViTConfig:
    return ViTConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_dim=mlp_dim,
        **kwargs,
    )


@register_model
def vit_tiny(
    image_size: int = 224, patch_size: int = 16, num_classes: int = 1000, **kwargs
) -> ViT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embedding_dim", "depth", "num_heads", "mlp_dim"},
        "factory variants do not allow overriding preset embedding_dim, depth, num_heads, or mlp_dim",
    )
    config = _build_vit_config(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_dim=192,
        depth=12,
        num_heads=3,
        mlp_dim=768,
        **kwargs,
    )
    return ViT(config)


@register_model
def vit_small(
    image_size: int = 224, patch_size: int = 16, num_classes: int = 1000, **kwargs
) -> ViT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embedding_dim", "depth", "num_heads", "mlp_dim"},
        "factory variants do not allow overriding preset embedding_dim, depth, num_heads, or mlp_dim",
    )
    config = _build_vit_config(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_dim=384,
        depth=12,
        num_heads=6,
        mlp_dim=1536,
        **kwargs,
    )
    return ViT(config)


@register_model
def vit_base(
    image_size: int = 224, patch_size: int = 16, num_classes: int = 1000, **kwargs
) -> ViT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embedding_dim", "depth", "num_heads", "mlp_dim"},
        "factory variants do not allow overriding preset embedding_dim, depth, num_heads, or mlp_dim",
    )
    config = _build_vit_config(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_dim=768,
        depth=12,
        num_heads=12,
        mlp_dim=3072,
        **kwargs,
    )
    return ViT(config)


@register_model
def vit_large(
    image_size: int = 224, patch_size: int = 16, num_classes: int = 1000, **kwargs
) -> ViT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embedding_dim", "depth", "num_heads", "mlp_dim"},
        "factory variants do not allow overriding preset embedding_dim, depth, num_heads, or mlp_dim",
    )
    config = _build_vit_config(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_dim=1024,
        depth=24,
        num_heads=16,
        mlp_dim=4096,
        **kwargs,
    )
    return ViT(config)


@register_model
def vit_huge(
    image_size: int = 224, patch_size: int = 16, num_classes: int = 1000, **kwargs
) -> ViT:
    _raise_for_locked_factory_kwargs(
        kwargs,
        {"embedding_dim", "depth", "num_heads", "mlp_dim"},
        "factory variants do not allow overriding preset embedding_dim, depth, num_heads, or mlp_dim",
    )
    config = _build_vit_config(
        image_size=image_size,
        patch_size=patch_size,
        num_classes=num_classes,
        embedding_dim=1280,
        depth=32,
        num_heads=16,
        mlp_dim=5120,
        **kwargs,
    )
    return ViT(config)
