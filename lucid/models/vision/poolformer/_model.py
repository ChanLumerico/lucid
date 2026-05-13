"""PoolFormer backbone and classifier (Yu et al., 2022).

Paper: "MetaFormer is Actually What You Need for Vision"

Key insight:
  The MetaFormer framework — norm → token-mixer → norm → channel-MLP, wrapped in
  residuals — drives performance regardless of the token-mixer choice.  Replacing
  multi-head self-attention with a plain average-pooling operation yields
  competitive accuracy at a fraction of the cost.

Architecture (PoolFormer-S12, embed_dims=(64,128,320,512), layers=(2,2,6,2)):
  PatchEmbed : Conv2d(3→64, 7×7, stride=4, pad=3) → GN(1,64)
  Stage 0    : 2 × PoolFormerBlock(64)
  Downsample : Conv2d(64→128, 3×3, stride=2, pad=1) → GN(1,128)
  Stage 1    : 2 × PoolFormerBlock(128)
  Downsample : Conv2d(128→320, 3×3, stride=2, pad=1) → GN(1,320)
  Stage 2    : 6 × PoolFormerBlock(320)
  Downsample : Conv2d(320→512, 3×3, stride=2, pad=1) → GN(1,512)
  Stage 3    : 2 × PoolFormerBlock(512)
  Head       : AdaptiveAvgPool2d(1,1) → flatten → LayerNorm → Linear

PoolFormer block (operates on (B, C, H, W)):
  x = x + LayerScale(PoolMixer(GN(1,C)(x)))
  x = x + LayerScale(MLP(GN(1,C)(x)))

PoolMixer: avg_pool(x, k=3, s=1, pad=1) − x
  (identity residual is handled by the outer + shortcut)

LayerScale: element-wise scale by a learnable (C,) parameter, broadcastto (B,C,H,W).

GroupNorm(1, C) ≡ LayerNorm applied to 2-D spatial feature maps — avoids the
need to permute to channel-last before normalisation.
"""

import lucid
from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.poolformer._config import PoolFormerConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _PoolMixer(nn.Module):
    """Pooling-based token mixer.

    Computes ``avg_pool(x) - x`` so that the plain residual connection
    ``x + token_mixer(x)`` becomes ``x + avg_pool(x) - x = avg_pool(x)``.
    """

    def __init__(self, pool_size: int = 3) -> None:
        super().__init__()
        self._pool_size = pool_size

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        pooled = F.avg_pool2d(
            x,
            kernel_size=self._pool_size,
            stride=1,
            padding=self._pool_size // 2,
        )
        return pooled - x


class _LayerScale(nn.Module):
    """Per-channel learnable scale factor, broadcast to (B, C, H, W)."""

    def __init__(self, dim: int, init: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(lucid.ones((dim,)) * init)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # scale: (C,) → (1, C, 1, 1) for broadcast with (B, C, H, W)
        return x * self.scale[None, :, None, None]


class _PoolFormerBlock(nn.Module):
    """Single PoolFormer block.

    x = x + LayerScale(PoolMixer(GN(x)))
    x = x + LayerScale(MLP(GN(x)))
    """

    def __init__(
        self,
        dim: int,
        pool_size: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        layer_scale_init: float = 1e-5,
    ) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)

        self.norm1 = nn.GroupNorm(1, dim)
        self.token_mixer = _PoolMixer(pool_size)
        self.ls1 = _LayerScale(dim, layer_scale_init)

        self.norm2 = nn.GroupNorm(1, dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden, dim, 1),
            nn.Dropout(p=dropout),
        )
        self.ls2 = _LayerScale(dim, layer_scale_init)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        normed1 = cast(Tensor, self.norm1(x))
        mixed1 = cast(Tensor, self.token_mixer(normed1))
        x = x + cast(Tensor, self.ls1(mixed1))
        normed2 = cast(Tensor, self.norm2(x))
        x = x + cast(Tensor, self.ls2(cast(Tensor, self.mlp(normed2))))
        return x


class _PatchEmbed(nn.Module):
    """Overlapping patch embedding via strided convolution."""

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        patch_size: int = 7,
        stride: int = 4,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            in_ch, embed_dim, patch_size, stride=stride, padding=patch_size // 2
        )
        self.norm = nn.GroupNorm(1, embed_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.norm(cast(Tensor, self.proj(x))))


class _Downsample(nn.Module):
    """2× spatial downsampling between stages via strided 3×3 conv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.norm = nn.GroupNorm(1, out_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.norm(cast(Tensor, self.proj(x))))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_poolformer(
    cfg: PoolFormerConfig,
) -> tuple[
    _PatchEmbed,
    nn.ModuleList,  # stages (each is nn.Sequential of _PoolFormerBlock)
    nn.ModuleList,  # downsamplers (len == num_stages - 1)
    nn.LayerNorm,  # head norm
    list[FeatureInfo],
    int,  # final embed_dim
]:
    patch_embed = _PatchEmbed(
        in_ch=cfg.in_channels,
        embed_dim=cfg.embed_dims[0],
        patch_size=7,
        stride=4,
    )

    stages: list[nn.Module] = []
    downsamplers: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4  # after patch embed (stride 4)

    num_stages = len(cfg.layers)
    for i in range(num_stages):
        dim = cfg.embed_dims[i]
        stage = nn.Sequential(
            *[
                _PoolFormerBlock(
                    dim=dim,
                    pool_size=cfg.pool_size,
                    mlp_ratio=cfg.mlp_ratio,
                    dropout=cfg.dropout,
                    layer_scale_init=cfg.layer_scale_init,
                )
                for _ in range(cfg.layers[i])
            ]
        )
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < num_stages - 1:
            next_dim = cfg.embed_dims[i + 1]
            downsamplers.append(_Downsample(dim, next_dim))
            reduction *= 2  # each downsample halves spatial dims

    head_norm = nn.LayerNorm(cfg.embed_dims[-1])

    return (
        patch_embed,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        head_norm,
        fi,
        cfg.embed_dims[-1],
    )


# ---------------------------------------------------------------------------
# PoolFormer backbone  (task="base")
# ---------------------------------------------------------------------------


class PoolFormer(PretrainedModel, BackboneMixin):
    """PoolFormer feature extractor — global-average-pooled final-stage features."""

    config_class: ClassVar[type[PoolFormerConfig]] = PoolFormerConfig
    base_model_prefix: ClassVar[str] = "poolformer"

    def __init__(self, config: PoolFormerConfig) -> None:
        super().__init__(config)
        patch_embed, stages, downs, hn, fi, out_dim = _build_poolformer(config)
        self.patch_embed = patch_embed
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._feature_info = fi
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.patch_embed(x))
        for i, stage in enumerate(self.stages):
            x = cast(Tensor, stage(x))
            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))
        # (B, C, H, W) → (B, C) via global avg pool + flatten
        x = cast(Tensor, self.avgpool(x)).flatten(1)
        return cast(Tensor, self.head_norm(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# PoolFormer for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class PoolFormerForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """PoolFormer with AdaptiveAvgPool + LayerNorm + FC classifier."""

    config_class: ClassVar[type[PoolFormerConfig]] = PoolFormerConfig
    base_model_prefix: ClassVar[str] = "poolformer"

    def __init__(self, config: PoolFormerConfig) -> None:
        super().__init__(config)
        patch_embed, stages, downs, hn, _, out_dim = _build_poolformer(config)
        self.patch_embed = patch_embed
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.patch_embed(x))
        for i, stage in enumerate(self.stages):
            x = cast(Tensor, stage(x))
            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))

        # Global average pool + flatten + head norm
        x = cast(Tensor, self.avgpool(x)).flatten(1)
        x = cast(Tensor, self.head_norm(x))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
