"""CrossViT backbone and classification head (Chen et al., 2021).

Two-branch Vision Transformer with different patch sizes. CLS tokens
cross-attend between branches to fuse multi-scale features.
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.crossvit._config import CrossViTConfig

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Non-overlapping patch embedding: Conv2d(patch_size, stride=patch_size)."""

    def __init__(self, in_ch: int, patch_size: int, dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.proj(x))
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, dim)


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.drop(F.gelu(cast(Tensor, self.fc1(x)))))
        return cast(Tensor, self.drop(cast(Tensor, self.fc2(x))))


class _SelfAttnBlock(nn.Module):
    """Standard ViT self-attention block."""

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, dropout: float
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        n = cast(Tensor, self.norm1(x))
        attn_out, _ = self.attn(n, n, n)
        x = x + attn_out
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Cross-attention module
# ---------------------------------------------------------------------------


class _CrossAttention(nn.Module):
    """Cross-attention from one branch's CLS token to another branch's tokens.

    The CLS token of the *source* branch attends over all tokens (incl. CLS)
    in the *target* branch. The attended result is added back to the source
    CLS position, enabling bidirectional information flow.
    """

    def __init__(self, dim_src: int, dim_tgt: int) -> None:
        super().__init__()
        self.norm_src = nn.LayerNorm(dim_src)
        self.norm_tgt = nn.LayerNorm(dim_tgt)
        # Project source CLS to target dim for cross-attention query
        self.q_proj = nn.Linear(dim_src, dim_tgt)
        self.attn = nn.MultiheadAttention(dim_tgt, num_heads=1, batch_first=True)
        # Project result back to source dim
        self.out_proj = nn.Linear(dim_tgt, dim_src)
        self.norm_out = nn.LayerNorm(dim_src)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:  # type: ignore[override]
        """Update src CLS token using cross-attention over tgt tokens.

        Args:
            src: (B, N_src, dim_src) — source branch tokens (CLS at index 0)
            tgt: (B, N_tgt, dim_tgt) — target branch tokens

        Returns:
            Updated (B, N_src, dim_src) source tokens.
        """
        cls_src = src[:, :1, :]  # (B, 1, dim_src)
        cls_normed = cast(Tensor, self.norm_src(cls_src))
        q = cast(Tensor, self.q_proj(cls_normed))  # (B, 1, dim_tgt)
        kv = cast(Tensor, self.norm_tgt(tgt))  # (B, N_tgt, dim_tgt)
        cross_out, _ = self.attn(q, kv, kv)  # (B, 1, dim_tgt)
        delta = cast(Tensor, self.norm_out(cast(Tensor, self.out_proj(cross_out))))
        new_cls = cls_src + delta
        return lucid.cat([new_cls, src[:, 1:, :]], dim=1)


# ---------------------------------------------------------------------------
# One CrossViT stage (self-attn per branch + cross-attn between branches)
# ---------------------------------------------------------------------------


class _CrossViTStage(nn.Module):
    def __init__(
        self,
        small_dim: int,
        large_dim: int,
        small_heads: int,
        large_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.small_block = _SelfAttnBlock(small_dim, small_heads, mlp_ratio, dropout)
        self.large_block = _SelfAttnBlock(large_dim, large_heads, mlp_ratio, dropout)
        # Cross-attention: small CLS → large tokens
        self.cross_s2l = _CrossAttention(small_dim, large_dim)
        # Cross-attention: large CLS → small tokens
        self.cross_l2s = _CrossAttention(large_dim, small_dim)

    def forward(  # type: ignore[override]
        self, x_s: Tensor, x_l: Tensor
    ) -> tuple[Tensor, Tensor]:
        x_s = cast(Tensor, self.small_block(x_s))
        x_l = cast(Tensor, self.large_block(x_l))
        # Exchange CLS information
        x_s = cast(Tensor, self.cross_s2l(x_s, x_l))
        x_l = cast(Tensor, self.cross_l2s(x_l, x_s))
        return x_s, x_l


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_trunk(
    config: CrossViTConfig,
) -> tuple[
    _PatchEmbed,  # small embed
    _PatchEmbed,  # large embed
    Tensor,  # small cls_token
    Tensor,  # large cls_token
    Tensor,  # small pos_embed
    Tensor,  # large pos_embed
    nn.Dropout,
    nn.ModuleList,  # stages
    nn.LayerNorm,  # small norm
    nn.LayerNorm,  # large norm
]:
    img = config.image_size
    n_s = (img // config.small_patch) ** 2
    n_l = (img // config.large_patch) ** 2

    embed_s = _PatchEmbed(config.in_channels, config.small_patch, config.small_dim)
    embed_l = _PatchEmbed(config.in_channels, config.large_patch, config.large_dim)

    cls_s = nn.Parameter(lucid.zeros(1, 1, config.small_dim))
    cls_l = nn.Parameter(lucid.zeros(1, 1, config.large_dim))
    pos_s = nn.Parameter(lucid.zeros(1, n_s + 1, config.small_dim))
    pos_l = nn.Parameter(lucid.zeros(1, n_l + 1, config.large_dim))
    nn.init.trunc_normal_(cls_s, std=0.02)
    nn.init.trunc_normal_(cls_l, std=0.02)
    nn.init.trunc_normal_(pos_s, std=0.02)
    nn.init.trunc_normal_(pos_l, std=0.02)

    drop = nn.Dropout(p=config.dropout)

    stages = nn.ModuleList(
        [
            _CrossViTStage(
                config.small_dim,
                config.large_dim,
                config.small_heads,
                config.large_heads,
                config.mlp_ratio,
                config.dropout,
            )
            for _ in range(config.depth)
        ]
    )
    norm_s = nn.LayerNorm(config.small_dim)
    norm_l = nn.LayerNorm(config.large_dim)

    return embed_s, embed_l, cls_s, cls_l, pos_s, pos_l, drop, stages, norm_s, norm_l


# ---------------------------------------------------------------------------
# CrossViT backbone (task="base")
# ---------------------------------------------------------------------------


class CrossViT(PretrainedModel, BackboneMixin):
    """CrossViT feature extractor — returns large branch CLS token.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` shaped
    ``(B, 1, large_dim)``.
    """

    config_class: ClassVar[type[CrossViTConfig]] = CrossViTConfig
    base_model_prefix: ClassVar[str] = "crossvit"

    cls_s: Tensor
    cls_l: Tensor
    pos_s: Tensor
    pos_l: Tensor

    def __init__(self, config: CrossViTConfig) -> None:
        super().__init__(config)
        (
            embed_s,
            embed_l,
            cls_s,
            cls_l,
            pos_s,
            pos_l,
            drop,
            stages,
            norm_s,
            norm_l,
        ) = _build_trunk(config)
        self.embed_s = embed_s
        self.embed_l = embed_l
        self.cls_s = cls_s
        self.cls_l = cls_l
        self.pos_s = pos_s
        self.pos_l = pos_l
        self.pos_drop = drop
        self.stages = stages
        self.norm_s = norm_s
        self.norm_l = norm_l
        self._feature_info = [
            FeatureInfo(
                stage=1, num_channels=config.large_dim, reduction=config.large_patch
            ),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        # Embed both branches
        x_s = cast(Tensor, self.embed_s(x))  # (B, N_s, small_dim)
        x_l = cast(Tensor, self.embed_l(x))  # (B, N_l, large_dim)
        # Prepend CLS tokens + add positional embedding
        x_s = lucid.cat([self.cls_s.expand(B, -1, -1), x_s], dim=1)
        x_l = lucid.cat([self.cls_l.expand(B, -1, -1), x_l], dim=1)
        x_s = cast(Tensor, self.pos_drop(x_s + self.pos_s))
        x_l = cast(Tensor, self.pos_drop(x_l + self.pos_l))
        # Cross-stage processing
        for stage in self.stages:
            result = cast(_CrossViTStage, stage).forward(x_s, x_l)
            x_s, x_l = result
        x_l = cast(Tensor, self.norm_l(x_l))
        return x_l[:, 0]  # large CLS: (B, large_dim)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# CrossViT for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class CrossViTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """CrossViT with dual-head classification using both branch CLS tokens."""

    config_class: ClassVar[type[CrossViTConfig]] = CrossViTConfig
    base_model_prefix: ClassVar[str] = "crossvit"

    cls_s: Tensor
    cls_l: Tensor
    pos_s: Tensor
    pos_l: Tensor

    def __init__(self, config: CrossViTConfig) -> None:
        super().__init__(config)
        (
            embed_s,
            embed_l,
            cls_s,
            cls_l,
            pos_s,
            pos_l,
            drop,
            stages,
            norm_s,
            norm_l,
        ) = _build_trunk(config)
        self.embed_s = embed_s
        self.embed_l = embed_l
        self.cls_s = cls_s
        self.cls_l = cls_l
        self.pos_s = pos_s
        self.pos_l = pos_l
        self.pos_drop = drop
        self.stages = stages
        self.norm_s = norm_s
        self.norm_l = norm_l
        # Paper §3.3: each branch has its *own* classifier; the final logits
        # are the *average* of both heads (not concat → FC).  ``classifier``
        # holds the small-branch head so :class:`ClassificationHeadMixin`'s
        # ``reset_classifier`` still works; ``head_l`` is the large-branch
        # head, kept in sync by our :meth:`reset_classifier` override below.
        self._build_classifier(config.small_dim, config.num_classes)
        self.head_l = nn.Linear(config.large_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        B = x.shape[0]
        x_s = cast(Tensor, self.embed_s(x))
        x_l = cast(Tensor, self.embed_l(x))
        x_s = lucid.cat([self.cls_s.expand(B, -1, -1), x_s], dim=1)
        x_l = lucid.cat([self.cls_l.expand(B, -1, -1), x_l], dim=1)
        x_s = cast(Tensor, self.pos_drop(x_s + self.pos_s))
        x_l = cast(Tensor, self.pos_drop(x_l + self.pos_l))
        for stage in self.stages:
            result = cast(_CrossViTStage, stage).forward(x_s, x_l)
            x_s, x_l = result
        x_s = cast(Tensor, self.norm_s(x_s))
        x_l = cast(Tensor, self.norm_l(x_l))
        # Paper §3.3: average the logits of two per-branch classifiers.
        logits_s = cast(Tensor, self.classifier(x_s[:, 0]))
        logits_l = cast(Tensor, self.head_l(x_l[:, 0]))
        logits = (logits_s + logits_l) * 0.5

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)

    def reset_classifier(self, num_classes: int) -> None:
        # Override to keep the large-branch head in sync with the small one.
        super().reset_classifier(num_classes)
        in_features = int(self.head_l.in_features)
        self.head_l = nn.Linear(in_features, num_classes)
