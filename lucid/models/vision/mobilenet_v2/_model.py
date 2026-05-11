"""MobileNet v2 backbone and classifier (Sandler et al., 2018).

Paper: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

Key idea: Inverted residual blocks — expand channels with 1×1 PW, apply
depthwise conv, then project back to a smaller channel count with a linear
(no-activation) 1×1 PW.  A residual shortcut is added only when stride==1
and input channels match output channels.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.mobilenet_v2._config import MobileNetV2Config


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# Inverted Residual block
# ---------------------------------------------------------------------------


class _InvertedResidual(nn.Module):
    """MobileNet v2 Inverted Residual (linear bottleneck) block."""

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
            # Depthwise
            nn.Conv2d(
                hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU6(inplace=True),
            # Pointwise linear (no activation)
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
# Architecture spec  (t=expand_ratio, c=out_ch, n=repeat, s=stride)
# ---------------------------------------------------------------------------

# (expand_ratio, out_channels, repeat, stride)
_INVERTED_RESIDUAL_SETTINGS: list[tuple[int, int, int, int]] = [
    (1, 16, 1, 1),
    (6, 24, 2, 2),
    (6, 32, 3, 2),
    (6, 64, 4, 2),
    (6, 96, 3, 1),
    (6, 160, 3, 2),
    (6, 320, 1, 1),
]


def _build_features(cfg: MobileNetV2Config) -> tuple[nn.Sequential, int]:
    """Return (features Sequential, num_out_channels)."""
    w = cfg.width_mult

    def _ch(c: int) -> int:
        return _make_divisible(c * w)

    # Stem
    stem_ch = _ch(32)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.ReLU6(inplace=True),
    ]

    in_ch = stem_ch
    for t, c, n, s in _INVERTED_RESIDUAL_SETTINGS:
        out_ch = _ch(c)
        for i in range(n):
            layers.append(
                _InvertedResidual(
                    in_ch, out_ch, stride=s if i == 0 else 1, expand_ratio=t
                )
            )
            in_ch = out_ch

    # Head conv
    last_ch = _ch(1280)
    layers += [
        nn.Conv2d(in_ch, last_ch, 1, bias=False),
        nn.BatchNorm2d(last_ch),
        nn.ReLU6(inplace=True),
    ]
    return nn.Sequential(*layers), last_ch


# ---------------------------------------------------------------------------
# MobileNet v2 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV2(PretrainedModel, BackboneMixin):
    """MobileNet v2 feature extractor."""

    config_class: ClassVar[type[MobileNetV2Config]] = MobileNetV2Config
    base_model_prefix: ClassVar[str] = "mobilenet_v2"

    def __init__(self, config: MobileNetV2Config) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult

        def _ch(c: int) -> int:
            return _make_divisible(c * w)

        cumulative = 1
        fi: list[FeatureInfo] = []
        for stage, (_, c, _, s) in enumerate(_INVERTED_RESIDUAL_SETTINGS):
            cumulative *= s
            fi.append(
                FeatureInfo(stage=stage + 1, num_channels=_ch(c), reduction=cumulative)
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
# MobileNet v2 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV2ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """MobileNet v2 with AdaptiveAvgPool + Dropout + FC classifier."""

    config_class: ClassVar[type[MobileNetV2Config]] = MobileNetV2Config
    base_model_prefix: ClassVar[str] = "mobilenet_v2"

    def __init__(self, config: MobileNetV2Config) -> None:
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
