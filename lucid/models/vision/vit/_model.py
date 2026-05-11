"""Vision Transformer (ViT) backbone and classifier (Dosovitskiy et al., 2020).

Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

Architecture:
    PatchEmbed  : Conv2d(patch_size, stride=patch_size) → flatten → Linear
    [CLS] token : prepended learnable token
    PosEmbed    : learnable (1, num_patches+1, dim)
    Dropout
    N × ViTBlock:
        LayerNorm → _Attention(qkv + proj) → residual
        LayerNorm → MLP(dim → mlp_dim → dim, GELU) → residual
    LayerNorm
    Head (backbone: CLS token; classifier: CLS → FC)
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.vit._config import ViTConfig

# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and linearly project each."""

    def __init__(self, in_channels: int, patch_size: int, dim: int) -> None:
        super().__init__()
        # A Conv2d with kernel=stride=patch_size extracts patches in one op.
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # (B, C, H, W) → (B, dim, H/p, W/p) → (B, num_patches, dim)
        x = cast(Tensor, self.proj(x))
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Self-attention with timm-compatible key naming
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Multi-head self-attention using a single fused qkv projection.

    Key naming matches timm's ViT:
        attn.qkv.weight  shape (3*dim, dim)
        attn.qkv.bias    shape (3*dim,)
        attn.proj.weight shape (dim, dim)
        attn.proj.bias   shape (dim,)
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Single fused projection for Q, K, V — matches timm naming exactly.
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # (B, N, 3*C) → (B, N, 3, H, D) → (3, B, H, N, D)
        qkv = cast(Tensor, self.qkv(x))
        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        # Each: (B, H, N, D)
        q = cast(Tensor, qkv[0])
        k = cast(Tensor, qkv[1])
        v = cast(Tensor, qkv[2])

        # Scaled dot-product attention
        attn = cast(Tensor, q @ k.permute(0, 1, 3, 2)) / self.scale
        attn = cast(Tensor, F.softmax(attn, dim=-1))
        attn = cast(Tensor, self.attn_drop(attn))

        # (B, H, N, D) → (B, N, H*D) = (B, N, C)
        out = cast(Tensor, attn @ v)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return cast(Tensor, self.proj(out))


# ---------------------------------------------------------------------------
# MLP inside each ViT block
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.drop(F.gelu(cast(Tensor, self.fc1(x)))))
        return cast(Tensor, self.drop(cast(Tensor, self.fc2(x))))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class _ViTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, attn_drop=attention_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x + cast(Tensor, self.attn(cast(Tensor, self.norm1(x))))
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_trunk(cfg: ViTConfig) -> tuple[
    _PatchEmbed,
    Tensor,  # cls_token parameter
    Tensor,  # pos_embed parameter
    nn.Dropout,
    nn.ModuleList,  # blocks
    nn.LayerNorm,  # norm
    int,  # num_patches
]:
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    mlp_dim = int(cfg.dim * cfg.mlp_ratio)

    patch_embed = _PatchEmbed(cfg.in_channels, cfg.patch_size, cfg.dim)

    cls_token = nn.Parameter(lucid.zeros(1, 1, cfg.dim))
    pos_embed = nn.Parameter(lucid.zeros(1, num_patches + 1, cfg.dim))
    # Simple uniform initialisation; no sinusoidal needed for learned pos embed
    nn.init.trunc_normal_(pos_embed, std=0.02)
    nn.init.trunc_normal_(cls_token, std=0.02)

    drop = nn.Dropout(p=cfg.dropout)
    blocks = nn.ModuleList(
        [
            _ViTBlock(
                cfg.dim, cfg.num_heads, mlp_dim, cfg.dropout, cfg.attention_dropout
            )
            for _ in range(cfg.depth)
        ]
    )
    norm = nn.LayerNorm(cfg.dim)

    return patch_embed, cls_token, pos_embed, drop, blocks, norm, num_patches


# ---------------------------------------------------------------------------
# ViT backbone  (task="base")
# ---------------------------------------------------------------------------


class ViT(PretrainedModel, BackboneMixin):
    """ViT feature extractor — returns (B, dim) CLS token after final LayerNorm.

    ``forward_features`` returns shape ``(B, dim)``.
    ``forward`` wraps it in ``BaseModelOutput`` (last_hidden_state shape (B, 1, dim)).
    """

    config_class: ClassVar[type[ViTConfig]] = ViTConfig
    base_model_prefix: ClassVar[str] = "vit"

    cls_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        pe, ct, pos, drop, blocks, norm, num_patches = _build_trunk(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.pos_embed = pos
        self.pos_drop = drop
        self.blocks = blocks
        self.norm = norm
        self._num_patches = num_patches
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=config.dim, reduction=config.patch_size),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = cast(Tensor, self.patch_embed(x))  # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = lucid.cat([cls, x], dim=1)  # (B, N+1, dim)
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))
        return x[:, 0]  # CLS token: (B, dim)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        # Unsqueeze to (B, 1, dim) so BaseModelOutput is spatially consistent
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# ViT for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ViTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """ViT with linear classification head on top of the CLS token.

    The classification head is ``self.head`` (a single ``nn.Linear``), matching
    timm's ``head.weight`` / ``head.bias`` state-dict key naming.
    """

    config_class: ClassVar[type[ViTConfig]] = ViTConfig
    base_model_prefix: ClassVar[str] = "vit"

    cls_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        pe, ct, pos, drop, blocks, norm, num_patches = _build_trunk(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.pos_embed = pos
        self.pos_drop = drop
        self.blocks = blocks
        self.norm = norm
        # Named ``head`` to match timm's vit_base_patch16_224 state-dict keys.
        self.head = nn.Linear(config.dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        B = x.shape[0]
        x = cast(Tensor, self.patch_embed(x))
        cls = self.cls_token.expand(B, -1, -1)
        x = lucid.cat([cls, x], dim=1)
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))
        logits = cast(Tensor, self.head(x[:, 0]))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
