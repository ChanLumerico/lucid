"""ConvNeXt backbone and classifier (Liu et al., 2022).

Paper: "A ConvNet for the 2020s"

Design recipe: systematically modernise ResNet by borrowing ideas from Swin:
  • Patchify stem (4×4 stride-4 conv, like ViT) instead of 7×7 stride-2
  • Stage ratios (3,3,9,3) matching Swin-T block counts
  • Inverted bottleneck MLP (4× expansion) borrowed from ViT
  • Depthwise 7×7 conv (large kernel, like shifted-window attention)
  • LayerNorm instead of BatchNorm; GELU instead of ReLU
  • Fewer normalisations (one LN per block, not BN-ReLU pairs)
  • Layer scale (per-channel γ initialised to 1e-6) for stable training

Architecture (ConvNeXt-T, dims=(96,192,384,768)):
  Stem     : Conv2d(4×4, stride=4) → LN                  → (56×56, 96)
  Stage 1  : 3 × ConvNeXtBlock(96)   → LN-Downsample(2×) → (28×28, 192)
  Stage 2  : 3 × ConvNeXtBlock(192)  → LN-Downsample(2×) → (14×14, 384)
  Stage 3  : 9 × ConvNeXtBlock(384)  → LN-Downsample(2×) → (7×7,  768)
  Stage 4  : 3 × ConvNeXtBlock(768)                       → (7×7,  768)
  Head     : AdaptiveAvgPool(1×1) → LN → FC

ConvNeXt block (all ops on (B, C, H, W)):
  DWConv(7×7, groups=C) → permute to (B,H,W,C)
  LN → Linear(C → 4C) → GELU → Linear(4C → C) → γ-scale
  permute back → + residual
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.convnext._config import ConvNeXtConfig

# ---------------------------------------------------------------------------
# ConvNeXt block
# ---------------------------------------------------------------------------


class _ConvNeXtBlock(nn.Module):
    """Depthwise 7×7 + inverted-bottleneck MLP + layer scale."""

    def __init__(self, dim: int, layer_scale_init: float) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(lucid.full((dim,), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = x
        x = cast(Tensor, self.dwconv(x))  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = cast(Tensor, self.norm(x))
        x = F.gelu(cast(Tensor, self.fc1(x)))
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
        # x: (B, C, H, W) → norm in channel-last → back → strided conv
        x = x.permute(0, 2, 3, 1)
        x = cast(Tensor, self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return cast(Tensor, self.conv(x))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_convnext(cfg: ConvNeXtConfig) -> tuple[
    _StemWithNorm,  # stem
    nn.ModuleList,  # stages
    nn.ModuleList,  # downsamplers (len = num_stages - 1)
    nn.LayerNorm,  # head norm
    list[FeatureInfo],
    int,  # final channels
]:
    # Patchify stem
    stem = nn.Sequential(
        nn.Conv2d(cfg.in_channels, cfg.dims[0], 4, stride=4),
        # LN applied in channel-last: use a wrapper
    )
    stem_norm = nn.LayerNorm(cfg.dims[0])

    stages: list[nn.Module] = []
    downsamplers: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4

    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
        stage = nn.Sequential(
            *[_ConvNeXtBlock(dim, cfg.layer_scale_init) for _ in range(depth)]
        )
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < len(cfg.depths) - 1:
            next_dim = cfg.dims[i + 1]
            downsamplers.append(_Downsample(dim, next_dim))
            reduction *= 2

    head_norm = nn.LayerNorm(cfg.dims[-1])

    # Wrap stem + stem_norm together
    full_stem = _StemWithNorm(stem, stem_norm)

    return (
        full_stem,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        head_norm,
        fi,
        cfg.dims[-1],
    )


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
# ConvNeXt backbone  (task="base")
# ---------------------------------------------------------------------------


class ConvNeXt(PretrainedModel, BackboneMixin):
    """ConvNeXt feature extractor — global-average-pooled final-stage features."""

    config_class: ClassVar[type[ConvNeXtConfig]] = ConvNeXtConfig
    base_model_prefix: ClassVar[str] = "convnext"

    stem: _StemWithNorm

    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, fi, out_dim = _build_convnext(config)
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
        x = x.unsqueeze(-1).unsqueeze(-1)  # keep (B,C,1,1) for consistency
        # Apply head norm in channel-last
        x = x.permute(0, 2, 3, 1)
        x = cast(Tensor, self.head_norm(x))
        return x.permute(0, 3, 1, 2).flatten(1)  # (B, C)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# ConvNeXt for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ConvNeXtForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """ConvNeXt with AdaptiveAvgPool + LN + FC classifier."""

    config_class: ClassVar[type[ConvNeXtConfig]] = ConvNeXtConfig
    base_model_prefix: ClassVar[str] = "convnext"

    stem: _StemWithNorm

    def __init__(self, config: ConvNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, _, out_dim = _build_convnext(config)
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
        # head norm in channel-last
        x = x.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 3, 1)
        x = cast(Tensor, self.head_norm(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
