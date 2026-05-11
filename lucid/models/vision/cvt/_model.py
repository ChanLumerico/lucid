"""CvT backbone and classification head (Wu et al., 2021).

Convolutional Vision Transformer: introduces overlapping convolutional token
embedding at each stage. Attention projections remain standard linear.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.cvt._config import CvTConfig

# ---------------------------------------------------------------------------
# Convolutional token embedding
# ---------------------------------------------------------------------------


class _ConvTokenEmbed(nn.Module):
    """Overlapping convolutional token embedding.

    Conv2d (stride=embed_stride, overlapping) → BN → GELU → flatten to sequence.
    This replaces the non-overlapping patch embedding of plain ViT.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 7,
        stride: int = 4,
        padding: int = 2,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # (B, C_in, H, W) → (B, out_ch, H', W') → (B, H'*W', out_ch)
        x = F.gelu(cast(Tensor, self.norm(cast(Tensor, self.proj(x)))))
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# MLP inside transformer block
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
# CvT transformer block
# ---------------------------------------------------------------------------


class _CvTBlock(nn.Module):
    """Pre-norm transformer block: LN → MHA → residual → LN → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
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
# CvT stage
# ---------------------------------------------------------------------------


class _CvTStage(nn.Module):
    """One CvT stage = ConvTokenEmbed + N × CvTBlock."""

    def __init__(
        self,
        in_ch: int,
        dim: int,
        depth: int,
        num_heads: int,
        embed_stride: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        # kernel=7 for first stage (stride=4), kernel=3 for subsequent stages (stride=2)
        kernel = 7 if embed_stride == 4 else 3
        padding = kernel // 2
        self.embed = _ConvTokenEmbed(
            in_ch, dim, kernel=kernel, stride=embed_stride, padding=padding
        )
        self.blocks = nn.ModuleList(
            [_CvTBlock(dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        """Accept (B, C, H, W) spatial feature map, return (B, N', dim)."""
        x = cast(Tensor, self.embed(x))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        return cast(Tensor, self.norm(x))


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_stages(config: CvTConfig) -> tuple[list[_CvTStage], list[FeatureInfo]]:
    stages: list[_CvTStage] = []
    in_ch = config.in_channels
    cum_stride = 1
    fi: list[FeatureInfo] = []
    for i, (dim, depth, heads, stride) in enumerate(
        zip(config.dims, config.depths, config.num_heads, config.embed_strides)
    ):
        stages.append(
            _CvTStage(
                in_ch, dim, depth, heads, stride, config.mlp_ratio, config.dropout
            )
        )
        in_ch = dim
        cum_stride *= stride
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=cum_stride))
    return stages, fi


# ---------------------------------------------------------------------------
# CvT backbone (task="base")
# ---------------------------------------------------------------------------


class CvT(PretrainedModel, BackboneMixin):
    """CvT feature extractor — returns mean-pooled token embedding.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` shaped ``(B, 1, dim)``
    (unsqueezed mean over tokens so the output is spatially consistent).
    """

    config_class: ClassVar[type[CvTConfig]] = CvTConfig
    base_model_prefix: ClassVar[str] = "cvt"

    def __init__(self, config: CvTConfig) -> None:
        super().__init__(config)
        stage_list, fi = _build_stages(config)
        self.stages = nn.ModuleList(stage_list)  # type: ignore[arg-type]
        self._feature_info = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def _forward_seq(self, x: Tensor) -> Tensor:
        """Pass through all stages; last stage output is (B, N, dim)."""
        # x starts as (B, C, H, W); first stage embed converts to (B, N, dim).
        # Subsequent stages: we need to convert (B, N, dim) back to (B, dim, H, W)
        # before feeding into the next stage's ConvTokenEmbed.
        # We track spatial dims through the stages.
        out = x
        for i, stage in enumerate(self.stages):
            if i == 0:
                out = cast(Tensor, stage(out))
            else:
                # Convert sequence back to spatial: we know hw from the config
                # But we don't store hw, so use the sequence length.
                B, N, C = out.shape
                import math

                H = W = int(math.isqrt(N))
                spatial = out.permute(0, 2, 1).reshape(B, C, H, W)
                out = cast(Tensor, stage(spatial))
        return out  # (B, N_final, dim_final)

    def forward_features(self, x: Tensor) -> Tensor:
        seq = self._forward_seq(x)
        return seq.mean(dim=1)  # (B, dim)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# CvT for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class CvTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """CvT with LayerNorm → mean pool → linear classification head."""

    config_class: ClassVar[type[CvTConfig]] = CvTConfig
    base_model_prefix: ClassVar[str] = "cvt"

    def __init__(self, config: CvTConfig) -> None:
        super().__init__(config)
        stage_list, _ = _build_stages(config)
        self.stages = nn.ModuleList(stage_list)  # type: ignore[arg-type]
        self.head_norm = nn.LayerNorm(config.dims[-1])
        self._build_classifier(
            config.dims[-1], config.num_classes, dropout=config.dropout
        )

    def _forward_seq(self, x: Tensor) -> Tensor:
        out = x
        for i, stage in enumerate(self.stages):
            if i == 0:
                out = cast(Tensor, stage(out))
            else:
                B, N, C = out.shape
                import math

                H = W = int(math.isqrt(N))
                spatial = out.permute(0, 2, 1).reshape(B, C, H, W)
                out = cast(Tensor, stage(spatial))
        return out

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        seq = self._forward_seq(x)
        seq = cast(Tensor, self.head_norm(seq))
        feat = seq.mean(dim=1)
        logits = cast(Tensor, self.classifier(feat))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
