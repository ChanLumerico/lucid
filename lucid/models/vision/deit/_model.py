"""DeiT backbone and classifier (Touvron et al., 2021).

Paper: "Training data-efficient image transformers & distillation through attention"

DeiT extends ViT with a distillation token — a second learnable token prepended
alongside the standard [cls] token.  The transformer blocks are identical to ViT.

Architecture:
  PatchEmbed → [cls | dist | patches] → PosEmbed → N × Block → LN
  cls_head  (from position 0 — the cls token)
  dist_head (from position 1 — the distillation token)
  Inference: logits = (cls_head(cls_out) + dist_head(dist_out)) / 2

Key dimensions:
  pos_embed shape: (1, num_patches + 2, dim)   — +2 for cls + dist
  cls_token  shape: (1, 1, dim)
  dist_token shape: (1, 1, dim)
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
from lucid.models.vision.deit._config import DeiTConfig

# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and linearly project each."""

    def __init__(self, in_channels: int, patch_size: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # (B, C, H, W) → (B, dim, H/p, W/p) → (B, num_patches, dim)
        x = cast(Tensor, self.proj(x))
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Self-attention
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Multi-head self-attention with fused qkv projection."""

    def __init__(self, dim: int, num_heads: int, attn_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # (B, N, 3*C) → (B, N, 3, H, D) → (3, B, H, N, D)
        qkv = cast(Tensor, self.qkv(x))
        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
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
# MLP inside each block
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
# Transformer block (identical to ViT)
# ---------------------------------------------------------------------------


class _Block(nn.Module):
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


def _build_trunk(cfg: DeiTConfig) -> tuple[
    _PatchEmbed,
    Tensor,         # cls_token parameter
    Tensor,         # dist_token parameter
    Tensor,         # pos_embed parameter
    nn.Dropout,
    nn.ModuleList,  # blocks
    nn.LayerNorm,   # norm
    int,            # num_patches
]:
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    mlp_dim = int(cfg.dim * cfg.mlp_ratio)

    patch_embed = _PatchEmbed(cfg.in_channels, cfg.patch_size, cfg.dim)

    cls_token = nn.Parameter(lucid.zeros(1, 1, cfg.dim))
    dist_token = nn.Parameter(lucid.zeros(1, 1, cfg.dim))
    # pos_embed covers cls + dist + patches: +2
    pos_embed = nn.Parameter(lucid.zeros(1, num_patches + 2, cfg.dim))

    nn.init.trunc_normal_(pos_embed, std=0.02)
    nn.init.trunc_normal_(cls_token, std=0.02)
    nn.init.trunc_normal_(dist_token, std=0.02)

    drop = nn.Dropout(p=cfg.dropout)
    blocks = nn.ModuleList(
        [
            _Block(cfg.dim, cfg.num_heads, mlp_dim, cfg.dropout, cfg.attention_dropout)
            for _ in range(cfg.depth)
        ]
    )
    norm = nn.LayerNorm(cfg.dim)

    return patch_embed, cls_token, dist_token, pos_embed, drop, blocks, norm, num_patches


# ---------------------------------------------------------------------------
# DeiT backbone  (task="base")
# ---------------------------------------------------------------------------


class DeiT(PretrainedModel, BackboneMixin):
    """DeiT feature extractor — returns (B, dim) CLS token after final LayerNorm.

    ``forward_features`` returns shape ``(B, dim)`` — the averaged output of
    the cls token and the distillation token after the final LayerNorm.
    ``forward`` wraps it in ``BaseModelOutput`` (last_hidden_state shape
    (B, 1, dim)).
    """

    config_class: ClassVar[type[DeiTConfig]] = DeiTConfig
    base_model_prefix: ClassVar[str] = "deit"

    cls_token: Tensor
    dist_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)
        pe, ct, dt, pos, drop, blocks, norm, num_patches = _build_trunk(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.dist_token = dt
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
        x = cast(Tensor, self.patch_embed(x))           # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, dim)
        dist = self.dist_token.expand(B, -1, -1)        # (B, 1, dim)
        x = lucid.cat([cls, dist, x], dim=1)            # (B, N+2, dim)
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))
        # Average of cls (position 0) and dist (position 1) tokens
        cls_out = cast(Tensor, x[:, 0])
        dist_out = cast(Tensor, x[:, 1])
        return (cls_out + dist_out) / 2  # (B, dim)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# DeiT for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class DeiTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """DeiT with two classification heads — one on cls token, one on dist token.

    At inference the logits from both heads are averaged, following the
    DeiT paper's evaluation protocol.
    """

    config_class: ClassVar[type[DeiTConfig]] = DeiTConfig
    base_model_prefix: ClassVar[str] = "deit"

    cls_token: Tensor
    dist_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: DeiTConfig) -> None:
        super().__init__(config)
        pe, ct, dt, pos, drop, blocks, norm, _ = _build_trunk(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.dist_token = dt
        self.pos_embed = pos
        self.pos_drop = drop
        self.blocks = blocks
        self.norm = norm
        # Two separate classification heads
        self.head = nn.Linear(config.dim, config.num_classes)
        self.head_dist = nn.Linear(config.dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        B = x.shape[0]
        x = cast(Tensor, self.patch_embed(x))
        cls = self.cls_token.expand(B, -1, -1)
        dist = self.dist_token.expand(B, -1, -1)
        x = lucid.cat([cls, dist, x], dim=1)            # (B, N+2, dim)
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))

        cls_out = cast(Tensor, x[:, 0])                  # (B, dim)
        dist_out = cast(Tensor, x[:, 1])                 # (B, dim)
        cls_logits = cast(Tensor, self.head(cls_out))
        dist_logits = cast(Tensor, self.head_dist(dist_out))
        # Average predictions from both heads (inference protocol)
        logits = (cls_logits + dist_logits) / 2

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
