"""MnasNet backbone and classifier (Tan et al., 2019).

Paper: "MnasNet: Platform-Aware Neural Architecture Search for Mobile"

Key ideas:
  1. MBConv (Mobile Inverted Bottleneck) blocks — same family as MobileNet V2.
  2. Squeeze-and-Excitation (SE) channel attention in select stages.
  3. Network topology searched via platform-aware NAS targeting mobile latency.

MnasNet-A1 baseline block specs (MnasNet-1.0, width_mult=1.0):
  Stem  : Conv2d(3, 32, 3, stride=2) → BN → ReLU
  Block1: _InvertedResidual(32→16,   k=3, s=1, e=1, n=1, se=0)
  Block2: _InvertedResidual(16→24,   k=3, s=2, e=6, n=2, se=0)
  Block3: _InvertedResidual(24→40,   k=5, s=2, e=3, n=3, se=0.25)
  Block4: _InvertedResidual(40→80,   k=3, s=2, e=6, n=4, se=0)
  Block5: _InvertedResidual(80→112,  k=3, s=1, e=6, n=2, se=0.25)
  Block6: _InvertedResidual(112→160, k=5, s=2, e=6, n=3, se=0.25)
  Block7: _InvertedResidual(160→320, k=3, s=1, e=6, n=1, se=0)
  Head  : Conv2d(320→1280, 1) → BN → ReLU → AdaptiveAvgPool2d(1,1)
  Classifier: Dropout → Linear(1280, num_classes)
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.mnasnet._config import MnasNetConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _round_ch(c: int, width_mult: float) -> int:
    return _make_divisible(c * width_mult)


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block (1×1 conv variant)
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention using 1×1 convolutions."""

    def __init__(self, in_channels: int, se_channels: int) -> None:
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        scale = cast(Tensor, self.squeeze(x))
        scale = F.relu(cast(Tensor, self.fc1(scale)))
        scale = F.sigmoid(cast(Tensor, self.fc2(scale)))
        return x * scale


# ---------------------------------------------------------------------------
# Inverted Residual (MBConv) block
# ---------------------------------------------------------------------------


class _InvertedResidual(nn.Module):
    """Mobile Inverted Bottleneck block with optional Squeeze-and-Excitation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        self._use_res = (stride == 1) and (in_ch == out_ch)
        hidden = in_ch * expand_ratio

        layers: list[nn.Module] = []

        # Expansion phase — skipped when expand_ratio == 1
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, hidden, 1, bias=False),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]

        # Depthwise conv
        layers += [
            nn.Conv2d(
                hidden,
                hidden,
                kernel,
                stride=stride,
                padding=kernel // 2,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        ]

        self.conv = nn.Sequential(*layers)

        # Squeeze-and-Excitation (applied to hidden/expanded channels)
        self.se: nn.Module
        if se_ratio > 0.0:
            se_ch = max(1, int(in_ch * se_ratio))
            self.se = _SEBlock(hidden, se_ch)
        else:
            self.se = nn.Identity()

        # Projection — no activation
        self.project_conv = nn.Conv2d(hidden, out_ch, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = cast(Tensor, self.conv(x))
        out = cast(Tensor, self.se(out))
        out = cast(Tensor, self.project_bn(cast(Tensor, self.project_conv(out))))
        if self._use_res:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Stage specification (MnasNet-A1 baseline, width_mult=1.0)
# (in_ch, out_ch, kernel, stride, expand_ratio, num_layers, se_ratio)
# ---------------------------------------------------------------------------

_BASE_SPECS: list[tuple[int, int, int, int, int, int, float]] = [
    (32, 16, 3, 1, 1, 1, 0.0),
    (16, 24, 3, 2, 6, 2, 0.0),
    (24, 40, 5, 2, 3, 3, 0.25),
    (40, 80, 3, 2, 6, 4, 0.0),
    (80, 112, 3, 1, 6, 2, 0.25),
    (112, 160, 5, 2, 6, 3, 0.25),
    (160, 320, 3, 1, 6, 1, 0.0),
]

_HEAD_CH = 1280


def _build_features(cfg: MnasNetConfig) -> tuple[nn.Sequential, int]:
    """Build the full feature extractor and return (Sequential, head_channels)."""
    w = cfg.width_mult

    stem_ch = _round_ch(32, w)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.ReLU(inplace=True),
    ]

    in_ch = stem_ch
    for base_in, base_out, kernel, stride, expand, n, se_ratio in _BASE_SPECS:
        out_ch = _round_ch(base_out, w)
        # First block in each stage uses the actual stride; subsequent blocks
        # are stride-1.  in_ch is tracked across all blocks.
        for i in range(n):
            layers.append(
                _InvertedResidual(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    kernel=kernel,
                    stride=stride if i == 0 else 1,
                    expand_ratio=expand,
                    se_ratio=se_ratio,
                )
            )
            in_ch = out_ch

    # Head conv: scale the 320 input, but keep 1280 output fixed
    head_in = _round_ch(320, w)
    # head_in matches in_ch at this point (last stage out_ch == round(320, w))
    layers += [
        nn.Conv2d(head_in, _HEAD_CH, 1, bias=False),
        nn.BatchNorm2d(_HEAD_CH),
        nn.ReLU(inplace=True),
    ]

    return nn.Sequential(*layers), _HEAD_CH


# ---------------------------------------------------------------------------
# MnasNet backbone  (task="base")
# ---------------------------------------------------------------------------


class MnasNet(PretrainedModel, BackboneMixin):
    """MnasNet feature extractor — global-average-pooled head features.

    Outputs (B, 1280, 1, 1) after AdaptiveAvgPool2d.
    """

    config_class: ClassVar[type[MnasNetConfig]] = MnasNetConfig
    base_model_prefix: ClassVar[str] = "mnasnet"

    def __init__(self, config: MnasNetConfig) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        # Build feature_info: cumulative stride at each stage output
        w = config.width_mult
        fi: list[FeatureInfo] = []
        cumulative = 2  # stem stride
        for i, (_, base_out, _, stride, _, _, _) in enumerate(_BASE_SPECS):
            cumulative *= stride
            fi.append(
                FeatureInfo(
                    stage=i + 1,
                    num_channels=_round_ch(base_out, w),
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
# MnasNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MnasNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """MnasNet with AdaptiveAvgPool + Dropout + FC classifier."""

    config_class: ClassVar[type[MnasNetConfig]] = MnasNetConfig
    base_model_prefix: ClassVar[str] = "mnasnet"

    def __init__(self, config: MnasNetConfig) -> None:
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
