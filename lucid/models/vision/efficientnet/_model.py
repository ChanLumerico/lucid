"""EfficientNet backbone and classifier (Tan & Le, 2019).

Paper: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"

Key ideas:
  1. MBConv (Mobile Inverted Bottleneck) blocks inherited from MobileNet v2.
  2. Squeeze-and-Excitation (SE) channel attention in each block.
  3. Compound scaling: width, depth, and resolution scaled simultaneously
     via a single compound coefficient φ.

B0 baseline block specs (expand_ratio, in_ch, out_ch, n_layers, stride, kernel):
  Stage 1: (1,  32,  16, 1, 1, 3)
  Stage 2: (6,  16,  24, 2, 2, 3)
  Stage 3: (6,  24,  40, 2, 2, 5)
  Stage 4: (6,  40,  80, 3, 2, 3)
  Stage 5: (6,  80, 112, 3, 1, 5)
  Stage 6: (6, 112, 192, 4, 2, 5)
  Stage 7: (6, 192, 320, 1, 1, 3)
  Head: Conv1×1(→1280) → AdaptiveAvgPool → Dropout → FC

Each MBConv block:
  1×1 expand conv (if expand_ratio > 1) → BN+SiLU
  k×k DW conv                            → BN+SiLU
  SE: squeeze → excite (sigmoid)
  1×1 project conv                       → BN  (no activation)
  + residual (if stride=1 & in==out)
"""

import math
from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.efficientnet._config import EfficientNetConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _round_channels(c: int, width_mult: float) -> int:
    return _make_divisible(c * width_mult)


def _round_layers(n: int, depth_mult: float) -> int:
    return max(1, math.ceil(n * depth_mult))


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        scale = cast(Tensor, self.squeeze(x))
        scale = F.silu(cast(Tensor, self.fc1(scale)))
        scale = F.sigmoid(cast(Tensor, self.fc2(scale)))
        return x * scale


# ---------------------------------------------------------------------------
# MBConv block
# ---------------------------------------------------------------------------


class _MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution with SE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float,
        drop_connect_rate: float,
    ) -> None:
        super().__init__()
        self._has_residual = (stride == 1) and (in_channels == out_channels)
        self._drop_connect_rate = drop_connect_rate

        expanded = in_channels * expand_ratio

        layers: list[nn.Module] = []
        # Expansion phase (skip if ratio == 1)
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
        se_channels = max(1, int(in_channels * se_ratio))
        self.se = _SEBlock(expanded, se_channels)

        # Projection
        self.project_conv = nn.Conv2d(expanded, out_channels, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = cast(Tensor, self.conv(x))
        out = cast(Tensor, self.se(out))
        out = cast(Tensor, self.project_bn(cast(Tensor, self.project_conv(out))))
        if self._has_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Block spec table and builder
# ---------------------------------------------------------------------------

# (expand_ratio, in_ch, out_ch, n_layers, stride, kernel_size)
_BASE_SPECS: list[tuple[int, int, int, int, int, int]] = [
    (1, 32, 16, 1, 1, 3),
    (6, 16, 24, 2, 2, 3),
    (6, 24, 40, 2, 2, 5),
    (6, 40, 80, 3, 2, 3),
    (6, 80, 112, 3, 1, 5),
    (6, 112, 192, 4, 2, 5),
    (6, 192, 320, 1, 1, 3),
]


def _build_features(cfg: EfficientNetConfig) -> tuple[nn.Sequential, int]:
    """Return (feature_extractor, num_final_channels)."""
    w = cfg.width_mult
    d = cfg.depth_mult

    stem_ch = _round_channels(32, w)
    stem = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.SiLU(inplace=True),
    ]

    # Count total blocks for linear drop_connect schedule
    total_blocks = sum(_round_layers(s[3], d) for s in _BASE_SPECS)
    block_idx = 0

    layers: list[nn.Module] = list(stem)
    in_ch = stem_ch

    for expand, base_in, base_out, base_n, stride, kernel in _BASE_SPECS:
        out_ch = _round_channels(base_out, w)
        n = _round_layers(base_n, d)
        for i in range(n):
            dc = cfg.drop_connect_rate * block_idx / max(1, total_blocks - 1)
            layers.append(
                _MBConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand,
                    se_ratio=cfg.se_ratio,
                    drop_connect_rate=dc,
                )
            )
            in_ch = out_ch
            block_idx += 1

    head_ch = _round_channels(1280, w)
    layers += [
        nn.Conv2d(in_ch, head_ch, 1, bias=False),
        nn.BatchNorm2d(head_ch),
        nn.SiLU(inplace=True),
    ]
    return nn.Sequential(*layers), head_ch


# ---------------------------------------------------------------------------
# EfficientNet backbone  (task="base")
# ---------------------------------------------------------------------------


class EfficientNet(PretrainedModel, BackboneMixin):
    """EfficientNet feature extractor — outputs (B, head_ch, 1, 1) after AvgPool."""

    config_class: ClassVar[type[EfficientNetConfig]] = EfficientNetConfig
    base_model_prefix: ClassVar[str] = "efficientnet"

    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult
        d = config.depth_mult
        strides = [1, 2, 2, 2, 1, 2, 1]
        cumulative = 1
        fi: list[FeatureInfo] = []
        for i, (_, _, base_out, _, stride, _) in enumerate(_BASE_SPECS):
            cumulative *= stride
            fi.append(
                FeatureInfo(
                    stage=i + 1,
                    num_channels=_round_channels(base_out, w),
                    reduction=cumulative,
                )
            )
        self._feature_info = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.features(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# EfficientNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class EfficientNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """EfficientNet with AdaptiveAvgPool + Dropout + FC classifier."""

    config_class: ClassVar[type[EfficientNetConfig]] = EfficientNetConfig
    base_model_prefix: ClassVar[str] = "efficientnet"

    def __init__(self, config: EfficientNetConfig) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
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
