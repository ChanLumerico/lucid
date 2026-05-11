"""MobileNet v4 backbone and classifier (Qin et al., 2024).

Paper: "MobileNetV4 - Universal Models for the Mobile Ecosystem"

Key idea: Universal Inverted Bottleneck (UIB) blocks generalise MBConv
by allowing optional depthwise convolutions at the start and middle of
each block.  This implementation approximates the Conv-Small variant
with standard inverted-residual (MBConv) blocks using the published
channel / stride specs.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.mobilenet_v4._config import MobileNetV4Config


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# Inverted Residual (MBConv) block — used for all UIB approximations
# ---------------------------------------------------------------------------


class _InvertedResidual(nn.Module):
    """Standard inverted-residual block (MBConv)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        expand_ratio: int,
    ) -> None:
        super().__init__()
        self._use_residual = stride == 1 and in_ch == out_ch
        hidden = in_ch * expand_ratio
        layers: list[nn.Module] = []

        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU6(inplace=True),
            ]

        layers += [
            nn.Conv2d(
                hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = cast(Tensor, self.conv(x))
        if self._use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Conv-Small architecture specs
# (out_ch, stride, expand_ratio)  — None expand_ratio = pointwise only
# ---------------------------------------------------------------------------

# Stem: Conv3×3 stride=2 → 32 ch
# Then inverted residual blocks
_CONV_SMALL_SPECS: list[tuple[int, int, int]] = [
    (32, 1, 2),
    (96, 2, 4),
    (64, 1, 1),
    (128, 2, 4),
    (128, 1, 1),
    (128, 1, 1),
    (256, 2, 4),
    (256, 1, 1),
    (256, 1, 1),
    (256, 1, 1),
    (256, 1, 1),
    (960, 1, 1),
    (1280, 1, 1),
]


def _build_features(cfg: MobileNetV4Config) -> tuple[nn.Sequential, int]:
    """Return (features, num_out_channels)."""
    stem_ch = 32
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.ReLU6(inplace=True),
    ]

    in_ch = stem_ch
    for out_ch, stride, expand_ratio in _CONV_SMALL_SPECS:
        if expand_ratio == 1 and in_ch == out_ch:
            # Pure depthwise-separable pass-through
            layers += [
                nn.Conv2d(
                    in_ch, in_ch, 3, stride=1, padding=1, groups=in_ch, bias=False
                ),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU6(inplace=True),
            ]
        else:
            layers.append(_InvertedResidual(in_ch, out_ch, stride, expand_ratio))
        in_ch = out_ch

    return nn.Sequential(*layers), in_ch


# ---------------------------------------------------------------------------
# MobileNet v4 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV4(PretrainedModel, BackboneMixin):
    """MobileNet v4 feature extractor (Conv-Small)."""

    config_class: ClassVar[type[MobileNetV4Config]] = MobileNetV4Config
    base_model_prefix: ClassVar[str] = "mobilenet_v4"

    def __init__(self, config: MobileNetV4Config) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        cumulative = 2  # stem stride=2
        fi: list[FeatureInfo] = []
        for i, (out_ch, s, _) in enumerate(_CONV_SMALL_SPECS):
            cumulative *= s
            fi.append(
                FeatureInfo(stage=i + 1, num_channels=out_ch, reduction=cumulative)
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
# MobileNet v4 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV4ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """MobileNet v4 with AdaptiveAvgPool + Dropout + FC classifier."""

    config_class: ClassVar[type[MobileNetV4Config]] = MobileNetV4Config
    base_model_prefix: ClassVar[str] = "mobilenet_v4"

    def __init__(self, config: MobileNetV4Config) -> None:
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
