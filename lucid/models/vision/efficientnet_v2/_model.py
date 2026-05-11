"""EfficientNetV2 backbone and classifier (Tan & Le, 2021).

Paper: "EfficientNetV2: Smaller Models and Faster Training"

Key difference from V1:
  Early stages use FusedMBConv (expand 3×3 conv + project 1×1 conv) instead of
  the DW-separable expand/DW/project pipeline.  Later stages retain standard
  MBConv with Squeeze-and-Excitation.

FusedMBConv:
  if expand_ratio == 1:
    Conv(in, out, k, stride) → BN → SiLU
  else:
    Conv(in, in*r, k, stride) → BN → SiLU
    Conv(in*r, out, 1)        → BN
  + residual when stride==1 and in_ch==out_ch

MBConv (identical to V1):
  [expand] Conv(in, in*r, 1) → BN → SiLU
  DW Conv(hidden, hidden, k, stride) → BN → SiLU
  SE: AvgPool → Conv(h, h//4) → SiLU → Conv(h//4, h) → Sigmoid
  proj Conv(hidden, out, 1) → BN
  + residual when stride==1 and in_ch==out_ch

Stochastic depth is omitted; the drop_connect_rate field in the config is
stored but not applied (inference-only framework path).
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.efficientnet_v2._config import EfficientNetV2Config

# ---------------------------------------------------------------------------
# Stage spec tables
# (block_type, expand_ratio, kernel, stride, in_ch, out_ch, num_blocks, se_ratio)
# ---------------------------------------------------------------------------

_StageSpec = tuple[str, int, int, int, int, int, int, float]

_STAGES_SMALL: list[_StageSpec] = [
    ("fused",  1, 3, 1,  24,  24,  2, 0.0),
    ("fused",  4, 3, 2,  24,  48,  4, 0.0),
    ("fused",  4, 3, 2,  48,  64,  4, 0.0),
    ("mbconv", 4, 3, 2,  64, 128,  6, 0.25),
    ("mbconv", 6, 3, 1, 128, 160,  9, 0.25),
    ("mbconv", 6, 3, 2, 160, 256, 15, 0.25),
]

_STAGES_MEDIUM: list[_StageSpec] = [
    ("fused",  1, 3, 1,  24,  24,  3, 0.0),
    ("fused",  4, 3, 2,  24,  48,  5, 0.0),
    ("fused",  4, 3, 2,  48,  80,  5, 0.0),
    ("mbconv", 4, 3, 2,  80, 160,  7, 0.25),
    ("mbconv", 6, 3, 1, 160, 176, 14, 0.25),
    ("mbconv", 6, 3, 2, 176, 304, 18, 0.25),
    ("mbconv", 6, 3, 1, 304, 512,  5, 0.25),
]

_STAGES_LARGE: list[_StageSpec] = [
    ("fused",  1, 3, 1,  32,  32,  4, 0.0),
    ("fused",  4, 3, 2,  32,  64,  7, 0.0),
    ("fused",  4, 3, 2,  64,  96,  7, 0.0),
    ("mbconv", 4, 3, 2,  96, 192, 10, 0.25),
    ("mbconv", 6, 3, 1, 192, 224, 19, 0.25),
    ("mbconv", 6, 3, 2, 224, 384, 25, 0.25),
    ("mbconv", 6, 3, 1, 384, 640,  7, 0.25),
]

_STAGES_XLARGE: list[_StageSpec] = [
    ("fused",  1, 3, 1,  32,  32,  4, 0.0),
    ("fused",  4, 3, 2,  32,  64,  8, 0.0),
    ("fused",  4, 3, 2,  64,  96,  8, 0.0),
    ("mbconv", 4, 3, 2,  96, 192, 16, 0.25),
    ("mbconv", 6, 3, 1, 192, 256, 24, 0.25),
    ("mbconv", 6, 3, 2, 256, 512, 32, 0.25),
    ("mbconv", 6, 3, 1, 512, 640,  8, 0.25),
]

_VARIANT_STAGES: dict[str, list[_StageSpec]] = {
    "small":  _STAGES_SMALL,
    "medium": _STAGES_MEDIUM,
    "large":  _STAGES_LARGE,
    "xlarge": _STAGES_XLARGE,
}

_HEAD_CH: int = 1280


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    """Channel-wise SE gate using 1×1 convolutions (4-D throughout)."""

    def __init__(self, in_channels: int, se_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        scale = cast(Tensor, self.pool(x))
        scale = F.silu(cast(Tensor, self.fc1(scale)))
        scale = F.sigmoid(cast(Tensor, self.fc2(scale)))
        return x * scale


# ---------------------------------------------------------------------------
# FusedMBConv block
# ---------------------------------------------------------------------------


class _FusedMBConvBlock(nn.Module):
    """Fused Mobile Inverted Bottleneck (no depthwise conv).

    When expand_ratio == 1 the block collapses to a single Conv → BN → SiLU.
    Otherwise: (expand 3×3 conv → BN → SiLU) → (project 1×1 conv → BN).
    A residual skip is added when stride == 1 and in_ch == out_ch.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self._has_residual: bool = (stride == 1) and (in_channels == out_channels)
        expanded: int = in_channels * expand_ratio

        if expand_ratio == 1:
            # Single fused conv (no separate project)
            self.fused_conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            )
            self.fused_bn = nn.BatchNorm2d(out_channels)
            self._no_expand: bool = True
        else:
            self.fused_conv = nn.Conv2d(
                in_channels,
                expanded,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            )
            self.fused_bn = nn.BatchNorm2d(expanded)
            self.proj_conv = nn.Conv2d(expanded, out_channels, 1, bias=False)
            self.proj_bn = nn.BatchNorm2d(out_channels)
            self._no_expand = False

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = F.silu(cast(Tensor, self.fused_bn(cast(Tensor, self.fused_conv(x)))))
        if not self._no_expand:
            out = cast(Tensor, self.proj_bn(cast(Tensor, self.proj_conv(out))))
        if self._has_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# MBConv block (V1-style, with SE)
# ---------------------------------------------------------------------------


class _MBConvBlock(nn.Module):
    """Standard Mobile Inverted Bottleneck with Squeeze-and-Excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
    ) -> None:
        super().__init__()
        self._has_residual: bool = (stride == 1) and (in_channels == out_channels)
        expanded: int = in_channels * expand_ratio

        layers: list[nn.Module] = []
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_channels, expanded, 1, bias=False),
                nn.BatchNorm2d(expanded),
                nn.SiLU(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(
                expanded,
                expanded,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expanded,
                bias=False,
            ),
            nn.BatchNorm2d(expanded),
            nn.SiLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

        # SE
        se_ch: int = max(1, int(in_channels * se_ratio))
        self.se = _SEBlock(expanded, se_ch)

        # Projection
        self.proj_conv = nn.Conv2d(expanded, out_channels, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = cast(Tensor, self.conv(x))
        out = cast(Tensor, self.se(out))
        out = cast(Tensor, self.proj_bn(cast(Tensor, self.proj_conv(out))))
        if self._has_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Feature extractor builder
# ---------------------------------------------------------------------------


def _build_features(
    cfg: EfficientNetV2Config,
) -> tuple[nn.Sequential, int, list[FeatureInfo]]:
    """Build stem + all stages + head conv.  Returns (features, head_ch, fi)."""
    stages = _VARIANT_STAGES[cfg.variant]

    # Stem uses the first stage's in_ch
    stem_in_ch: int = stages[0][4]

    stem_layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_in_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_in_ch),
        nn.SiLU(inplace=True),
    ]
    all_layers: list[nn.Module] = list(stem_layers)

    fi: list[FeatureInfo] = []
    cumulative_stride: int = 2  # stem is stride-2

    for stage_idx, (blk_type, expand, kernel, stride, in_ch, out_ch, n_blks, se_r) in enumerate(stages):
        for i in range(n_blks):
            s: int = stride if i == 0 else 1
            ic: int = in_ch if i == 0 else out_ch

            if blk_type == "fused":
                all_layers.append(
                    _FusedMBConvBlock(ic, out_ch, kernel, s, expand)
                )
            else:
                all_layers.append(
                    _MBConvBlock(ic, out_ch, kernel, s, expand, se_r)
                )

        if stride > 1:
            cumulative_stride *= stride
        fi.append(
            FeatureInfo(
                stage=stage_idx + 1,
                num_channels=out_ch,
                reduction=cumulative_stride,
            )
        )

    last_ch: int = stages[-1][5]  # out_ch of final stage
    all_layers += [
        nn.Conv2d(last_ch, _HEAD_CH, 1, bias=False),
        nn.BatchNorm2d(_HEAD_CH),
        nn.SiLU(inplace=True),
    ]

    return nn.Sequential(*all_layers), _HEAD_CH, fi


# ---------------------------------------------------------------------------
# EfficientNetV2 backbone (task="base")
# ---------------------------------------------------------------------------


class EfficientNetV2(PretrainedModel, BackboneMixin):
    """EfficientNetV2 feature extractor — outputs pooled (B, 1280, 1, 1)."""

    config_class: ClassVar[type[EfficientNetV2Config]] = EfficientNetV2Config
    base_model_prefix: ClassVar[str] = "efficientnet_v2"

    def __init__(self, config: EfficientNetV2Config) -> None:
        super().__init__(config)
        features, num_features, fi = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features: int = num_features
        self._feature_info: list[FeatureInfo] = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.features(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# EfficientNetV2 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class EfficientNetV2ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """EfficientNetV2 with AdaptiveAvgPool + Dropout + FC classifier."""

    config_class: ClassVar[type[EfficientNetV2Config]] = EfficientNetV2Config
    base_model_prefix: ClassVar[str] = "efficientnet_v2"

    def __init__(self, config: EfficientNetV2Config) -> None:
        super().__init__(config)
        features, num_features, _ = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(num_features, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.features(x))
        x = cast(Tensor, self.avgpool(x))
        x = cast(Tensor, self.drop(x.flatten(1)))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
