import lucid
import lucid.nn as nn
import lucid.nn.functional as F

from lucid import register_model
from lucid._tensor import Tensor


__all__ = ["ViT", "vit-s", "vit_b", "vit_l", "vit_h", "vit-g"]


class ViT(nn.Module):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embedding_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_dim: int = 3072,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size == 0:
            raise ValueError("image_size must be divisible by the patch_size.")

        self.embedding_dim = embedding_dim
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_emb = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )

        self.cls_token = nn.Parameter(lucid.random.randn(1, 1, embedding_dim))
        self.pos_emb = nn.Parameter(
            lucid.random.randn(1, 1 + self.num_patches, embedding_dim)
        )
        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            num_heads=num_heads,
            dim_feedforward=mlp_dim,
            dropout=dropout_rate,
            activation=F.gelu,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=depth
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

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


@register_model
def vit_s(patch_size: int = 16, num_classes: int = 1000, **kwargs) -> ViT:
    NotImplemented


@register_model
def vit_b(patch_size: int = 16, num_classes: int = 1000, **kwargs) -> ViT:
    NotImplemented


@register_model
def vit_l(patch_size: int = 16, num_classes: int = 1000, **kwargs) -> ViT:
    NotImplemented


@register_model
def vit_h(patch_size: int = 16, num_classes: int = 1000, **kwargs) -> ViT:
    NotImplemented


@register_model
def vit_g(patch_size: int = 16, num_classes: int = 1000, **kwargs) -> ViT:
    NotImplemented
