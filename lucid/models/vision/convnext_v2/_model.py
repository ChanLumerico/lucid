"""ConvNeXt V2 backbone and classifier (Woo et al., 2022).

Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"

ConvNeXt V2 extends ConvNeXt V1 with one key addition: Global Response
Normalization (GRN) inserted inside each MLP block after the GELU activation.

ConvNeXt V2 block:
  DWConv(7×7, groups=C) → permute to (B,H,W,C)
  LN → Linear(C → 4C) → GELU → GRN → Linear(4C → C) → γ-scale
  permute back → + residual

GRN (Global Response Normalization):
  Given x of shape (B, H, W, C):
    gx = L2 norm of x over spatial dims (H, W), shape (B, 1, 1, C)
    nx = gx / (mean(gx over channels) + ε)
    out = γ * (x * nx) + β + x
  where γ and β are per-channel learnable scalars initialised to 0.

Architecture (same macro-structure as ConvNeXt V1):
  Stem     : Conv2d(4×4, stride=4) → LN                  → (56×56, dims[0])
  Stage i  : depth[i] × ConvNeXtV2Block(dims[i])
  Between stages : LN-Downsample (2×2 strided conv)
  Head     : AdaptiveAvgPool(1×1) → LN → FC
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.convnext_v2._config import ConvNeXtV2Config

# ---------------------------------------------------------------------------
# Global Response Normalization
# ---------------------------------------------------------------------------


class _GRN(nn.Module):
    """Global Response Normalization (GRN) layer.

    Operates on tensors of shape (B, H, W, C) — channel-last layout.
    All parameters (gamma, beta) are initialised to zero.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(lucid.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(lucid.zeros(1, 1, 1, dim))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, H, W, C)
        # Compute L2 norm over spatial dims (1, 2), keepdim → (B, 1, 1, C)
        gx: Tensor = (x * x).sum(dim=(1, 2), keepdim=True)
        gx = (gx + 1e-6) ** 0.5
        # Normalise across channels: divide by mean over C dim
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


# ---------------------------------------------------------------------------
# ConvNeXt V2 block
# ---------------------------------------------------------------------------


class _ConvNeXtV2Block(nn.Module):
    """Depthwise 7×7 + inverted-bottleneck MLP + GRN + layer scale."""

    def __init__(self, dim: int, layer_scale_init: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.grn = _GRN(4 * dim)
        self.gamma = nn.Parameter(lucid.full((dim,), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = x
        x = cast(Tensor, self.dwconv(x))  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = cast(Tensor, self.norm(x))
        x = cast(Tensor, self.fc1(x))
        x = F.gelu(x)
        x = cast(Tensor, self.grn(x))  # GRN after GELU, on 4C features
        x = cast(Tensor, self.fc2(x))
        x = x * self.gamma  # layer scale
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return shortcut + x


# ---------------------------------------------------------------------------
# Downsampling between stages
# ---------------------------------------------------------------------------


class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Norm in channel-last then conv in channel-first
        x = x.permute(0, 2, 3, 1)
        x = cast(Tensor, self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return cast(Tensor, self.conv(x))


# ---------------------------------------------------------------------------
# Patchify stem with LayerNorm
# ---------------------------------------------------------------------------


class _StemWithNorm(nn.Module):
    def __init__(self, conv: nn.Sequential, norm: nn.LayerNorm) -> None:
        super().__init__()
        self.conv = conv
        self.norm = norm

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.conv(x))  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = cast(Tensor, self.norm(x))
        return x.permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_convnext_v2(cfg: ConvNeXtV2Config) -> tuple[
    _StemWithNorm,
    nn.ModuleList,  # stages
    nn.ModuleList,  # downsamplers
    nn.LayerNorm,  # head norm
    list[FeatureInfo],
    int,  # final channels
]:
    stem_conv = nn.Sequential(
        nn.Conv2d(cfg.in_channels, cfg.dims[0], 4, stride=4),
    )
    stem_norm = nn.LayerNorm(cfg.dims[0])
    full_stem = _StemWithNorm(stem_conv, stem_norm)

    stages: list[nn.Module] = []
    downsamplers: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4

    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
        stage = nn.Sequential(
            *[_ConvNeXtV2Block(dim, cfg.layer_scale_init) for _ in range(depth)]
        )
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < len(cfg.depths) - 1:
            next_dim = cfg.dims[i + 1]
            downsamplers.append(_Downsample(dim, next_dim))
            reduction *= 2

    head_norm = nn.LayerNorm(cfg.dims[-1])

    return (
        full_stem,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        head_norm,
        fi,
        cfg.dims[-1],
    )


# ---------------------------------------------------------------------------
# ConvNeXt V2 backbone  (task="base")
# ---------------------------------------------------------------------------


class ConvNeXtV2(PretrainedModel, BackboneMixin):
    """ConvNeXt V2 feature extractor — global-average-pooled final-stage features."""

    config_class: ClassVar[type[ConvNeXtV2Config]] = ConvNeXtV2Config
    base_model_prefix: ClassVar[str] = "convnext_v2"

    stem: _StemWithNorm

    def __init__(self, config: ConvNeXtV2Config) -> None:
        super().__init__(config)
        stem, stages, downs, hn, fi, out_dim = _build_convnext_v2(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._feature_info = fi
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        for i, stage in enumerate(self.stages):
            x = cast(Tensor, stage(x))
            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))
        x = cast(Tensor, self.avgpool(x)).flatten(1)  # (B, C)
        # Apply head norm in channel-last
        x = x.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        x = x.permute(0, 2, 3, 1)  # (B, 1, 1, C)
        x = cast(Tensor, self.head_norm(x))
        return x.permute(0, 3, 1, 2).flatten(1)  # (B, C)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# ConvNeXt V2 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ConvNeXtV2ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """ConvNeXt V2 with AdaptiveAvgPool + LN + FC classifier."""

    config_class: ClassVar[type[ConvNeXtV2Config]] = ConvNeXtV2Config
    base_model_prefix: ClassVar[str] = "convnext_v2"

    stem: _StemWithNorm

    def __init__(self, config: ConvNeXtV2Config) -> None:
        super().__init__(config)
        stem, stages, downs, hn, _, out_dim = _build_convnext_v2(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        for i, stage in enumerate(self.stages):
            x = cast(Tensor, stage(x))
            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))
        x = cast(Tensor, self.avgpool(x)).flatten(1)
        # Head norm in channel-last
        x = x.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 3, 1)
        x = cast(Tensor, self.head_norm(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
