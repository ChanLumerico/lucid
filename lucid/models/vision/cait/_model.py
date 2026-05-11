"""CaiT backbone and classifier (Touvron et al., 2021).

Paper: "Going deeper with Image Transformers"

Key innovations vs ViT:
  - LayerScale: per-channel learnable scaling (γ) added to each residual
    branch; initialised to a small value (1e-5 or 1e-6) for training
    stability in very deep networks.
  - Class Attention (CA) layers: after N self-attention blocks on the patch
    tokens, 2 CA blocks insert a class token.  In CA, *only* the class token
    queries the full sequence (cls + patches); patch tokens are not updated.

Architecture (CaiT-XXS/24, dim=192, depth=24, heads=4):
  PatchEmbed   : Conv2d(patch_size, stride=patch_size) → (B, N, dim)
  pos_embed    : (1, N, dim) — no position on cls token
  N × SelfAttnBlock : standard MHSA + MLP, each with LayerScale
  prepend cls_token
  2 × ClassAttnBlock: class attention + MLP, each with LayerScale
  LayerNorm → cls_token[:, 0] → head
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
from lucid.models.vision.cait._config import CaiTConfig


# ---------------------------------------------------------------------------
# Patch embedding (shared with ViT pattern)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Non-overlapping patch embedding via a single strided Conv2d."""

    def __init__(self, in_channels: int, patch_size: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.proj(x))  # (B, dim, H/p, W/p)
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, dim)


# ---------------------------------------------------------------------------
# LayerScale
# ---------------------------------------------------------------------------


class _LayerScale(nn.Module):
    """Per-channel learnable scale applied to a residual branch.

    γ is initialised to ``init_value`` (e.g. 1e-5) so the residual branch
    starts near zero, giving effective identity-like behaviour at the start
    of training.  This allows very deep networks to train stably without
    careful initialisation of the attention weights.

    Reference: Touvron et al. (2021), §3.1.
    """

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(lucid.full((dim,), init_value))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N, C) — broadcast gamma over B and N
        return x * self.gamma


# ---------------------------------------------------------------------------
# MLP used inside both block types
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
# Standard multi-head self-attention (for patch tokens)
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Multi-head self-attention using a fused qkv projection."""

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

        qkv = cast(Tensor, self.qkv(x))                      # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q: Tensor = qkv[0]                                    # (B, H, N, D)
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]

        attn = q @ k.permute(0, 1, 3, 2) / self.scale        # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = cast(Tensor, self.attn_drop(attn))

        out = attn @ v                                        # (B, H, N, D)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return cast(Tensor, self.proj(out))


# ---------------------------------------------------------------------------
# Class Attention: only the cls token queries the full sequence
# ---------------------------------------------------------------------------


class _ClassAttention(nn.Module):
    """Class attention layer: Q from cls token, K/V from all tokens.

    In CaiT's class-attention stage, the cls token is the only query.
    Patch tokens do not receive updates — only cls is modified.

    Reference: Touvron et al. (2021), §3.2.
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.k = nn.Linear(dim, dim, bias=True)
        self.v = nn.Linear(dim, dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N+1, C) where x[:, 0] is the cls token
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # Q: cls token only → (B, H, 1, D)
        cls_in: Tensor = x[:, :1]
        q = cast(Tensor, self.q(cls_in)).reshape(B, 1, H, D).permute(0, 2, 1, 3)
        # K, V: all tokens → (B, H, N, D)
        k = cast(Tensor, self.k(x)).reshape(B, N, H, D).permute(0, 2, 1, 3)
        v = cast(Tensor, self.v(x)).reshape(B, N, H, D).permute(0, 2, 1, 3)

        attn = q @ k.permute(0, 1, 3, 2) / self.scale  # (B, H, 1, N)
        attn = F.softmax(attn, dim=-1)
        attn = cast(Tensor, self.attn_drop(attn))

        out = attn @ v                              # (B, H, 1, D)
        out = out.permute(0, 2, 1, 3).reshape(B, 1, C)
        return cast(Tensor, self.proj(out))          # (B, 1, C)


# ---------------------------------------------------------------------------
# Self-attention block (for patch tokens)
# ---------------------------------------------------------------------------


class _SelfAttnBlock(nn.Module):
    """Standard ViT-style block with LayerScale on both residual branches."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_drop: float,
        ls_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, attn_drop)
        self.ls1 = _LayerScale(dim, ls_init)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_dim, dropout)
        self.ls2 = _LayerScale(dim, ls_init)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N, C) — patch tokens only
        x = x + cast(Tensor, self.drop(cast(Tensor, self.ls1(cast(Tensor, self.attn(cast(Tensor, self.norm1(x))))))))
        x = x + cast(Tensor, self.drop(cast(Tensor, self.ls2(cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))))))
        return x


# ---------------------------------------------------------------------------
# Class attention block
# ---------------------------------------------------------------------------


class _ClassAttnBlock(nn.Module):
    """Class-attention block: cls queries all tokens; patches are not updated.

    The block signature takes (patches, cls) separately and returns both.
    Only cls is updated (both by class attention and by the MLP).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_drop: float,
        ls_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _ClassAttention(dim, num_heads, attn_drop)
        self.ls1 = _LayerScale(dim, ls_init)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, mlp_dim, dropout)
        self.ls2 = _LayerScale(dim, ls_init)

    def forward(self, x: Tensor, cls: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        # x:   (B, N, C)   — patch tokens (unchanged by this block)
        # cls: (B, 1, C)   — class token
        # Concatenate for LN and attention input
        xc = lucid.cat([cls, x], dim=1)             # (B, N+1, C)
        xc_norm = cast(Tensor, self.norm1(xc))
        cls = cls + cast(Tensor, self.ls1(cast(Tensor, self.attn(xc_norm))))   # (B, 1, C)
        cls = cls + cast(Tensor, self.ls2(cast(Tensor, self.mlp(cast(Tensor, self.norm2(cls))))))
        return x, cls


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_cait(cfg: CaiTConfig) -> tuple[
    _PatchEmbed,
    Tensor,           # cls_token parameter
    Tensor,           # pos_embed parameter
    nn.Dropout,
    nn.ModuleList,    # self-attention blocks
    nn.ModuleList,    # class-attention blocks
    nn.LayerNorm,     # final norm
    int,              # num_patches
]:
    num_patches = (cfg.image_size // cfg.patch_size) ** 2

    patch_embed = _PatchEmbed(cfg.in_channels, cfg.patch_size, cfg.dim)

    cls_token: Tensor = nn.Parameter(lucid.zeros(1, 1, cfg.dim))
    # pos_embed covers only patch tokens (no cls position)
    pos_embed: Tensor = nn.Parameter(lucid.zeros(1, num_patches, cfg.dim))
    nn.init.trunc_normal_(pos_embed, std=0.02)
    nn.init.trunc_normal_(cls_token, std=0.02)

    pos_drop = nn.Dropout(p=cfg.dropout)

    sa_blocks = nn.ModuleList([
        _SelfAttnBlock(
            cfg.dim, cfg.num_heads, cfg.mlp_ratio,
            cfg.dropout, cfg.attention_dropout, cfg.layer_scale_init,
        )
        for _ in range(cfg.depth)
    ])

    ca_blocks = nn.ModuleList([
        _ClassAttnBlock(
            cfg.dim, cfg.num_heads, cfg.mlp_ratio,
            cfg.dropout, cfg.attention_dropout, cfg.layer_scale_init,
        )
        for _ in range(cfg.class_depth)
    ])

    norm = nn.LayerNorm(cfg.dim)
    return patch_embed, cls_token, pos_embed, pos_drop, sa_blocks, ca_blocks, norm, num_patches


# ---------------------------------------------------------------------------
# CaiT backbone
# ---------------------------------------------------------------------------


class CaiT(PretrainedModel, BackboneMixin):
    """CaiT feature extractor — returns cls token (B, dim) after final LayerNorm.

    ``forward_features`` returns shape ``(B, dim)``.
    ``forward`` wraps it in ``BaseModelOutput``.
    """

    config_class: ClassVar[type[CaiTConfig]] = CaiTConfig
    base_model_prefix: ClassVar[str] = "cait"

    cls_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: CaiTConfig) -> None:
        super().__init__(config)
        pe, ct, pos, drop, sa, ca, norm, num_patches = _build_cait(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.pos_embed = pos
        self.pos_drop = drop
        self.blocks = sa
        self.class_blocks = ca
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
        x = cast(Tensor, self.patch_embed(x))         # (B, N, dim)
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))

        # Self-attention stage — patch tokens only
        for blk in self.blocks:
            x = cast(Tensor, blk(x))

        # Class-attention stage — insert cls token
        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, dim)
        for blk in self.class_blocks:
            x, cls = blk(x, cls)

        # Normalise and return cls token
        cls = cast(Tensor, self.norm(cls))
        cls_out: Tensor = cls[:, 0]                    # (B, dim)
        return cls_out

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# CaiT for image classification
# ---------------------------------------------------------------------------


class CaiTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """CaiT with linear classification head on top of the cls token."""

    config_class: ClassVar[type[CaiTConfig]] = CaiTConfig
    base_model_prefix: ClassVar[str] = "cait"

    cls_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: CaiTConfig) -> None:
        super().__init__(config)
        pe, ct, pos, drop, sa, ca, norm, _ = _build_cait(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.pos_embed = pos
        self.pos_drop = drop
        self.blocks = sa
        self.class_blocks = ca
        self.norm = norm
        self.head = nn.Linear(config.dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        B = x.shape[0]
        x = cast(Tensor, self.patch_embed(x))
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))

        for blk in self.blocks:
            x = cast(Tensor, blk(x))

        cls = self.cls_token.expand(B, -1, -1)
        for blk in self.class_blocks:
            x, cls = blk(x, cls)

        cls = cast(Tensor, self.norm(cls))
        cls_tok: Tensor = cls[:, 0]
        logits = cast(Tensor, self.head(cls_tok))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
