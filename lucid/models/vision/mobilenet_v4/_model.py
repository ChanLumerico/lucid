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
# Architecture specs per variant: (out_ch, stride, expand_ratio)
# Stem is always Conv3×3 stride=2 (32 ch for small/medium, 24 ch for large).
# ---------------------------------------------------------------------------

# Conv-Small (~3.7 M with the simplified MBConv blocks used here)
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

# Conv-Medium (~9.7 M).  Stem → 32 ch.
# Reflects the published channel/stride schedule from Qin et al. 2024,
# mapped onto MBConv blocks (expand_ratio approximates UIB expansion).
_CONV_MEDIUM_SPECS: list[tuple[int, int, int]] = [
    # layer1: fused-IB 32→48 stride 2, expand 4
    (48, 2, 4),
    # layer2: 48→80 stride 2, 80→80 stride 1
    (80, 2, 4),
    (80, 1, 2),
    # layer3: 8 blocks  (80→160 stride 2, then 7× 160 stride 1)
    (160, 2, 6),
    (160, 1, 4),
    (160, 1, 4),
    (160, 1, 4),
    (160, 1, 4),
    (160, 1, 4),
    (160, 1, 2),
    (160, 1, 4),
    # layer4: 11 blocks (160→256 stride 2, then 10× 256 stride 1)
    (256, 2, 6),
    (256, 1, 4),
    (256, 1, 4),
    (256, 1, 4),
    (256, 1, 4),
    (256, 1, 4),
    (256, 1, 2),
    (256, 1, 4),
    (256, 1, 4),
    (256, 1, 4),
    (256, 1, 2),
    # layer5: pointwise head
    (960, 1, 1),
    (1280, 1, 1),
]

# Conv-Large (~32.6 M).  Stem → 24 ch.
_CONV_LARGE_SPECS: list[tuple[int, int, int]] = [
    # layer1: fused-IB 24→48 stride 2, expand 4
    (48, 2, 4),
    # layer2: 48→96 stride 2, 96→96 stride 1
    (96, 2, 4),
    (96, 1, 4),
    # layer3: 11 blocks (96→192 stride 2, then 10× 192 stride 1)
    (192, 2, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    (192, 1, 4),
    # layer4: 13 blocks (192→512 stride 2, then 12× 512 stride 1)
    (512, 2, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    (512, 1, 4),
    # layer5: pointwise head
    (960, 1, 1),
    (1280, 1, 1),
]

_VARIANT_SPECS: dict[str, tuple[int, list[tuple[int, int, int]]]] = {
    "conv_small": (32, _CONV_SMALL_SPECS),
    "conv_medium": (32, _CONV_MEDIUM_SPECS),
    "conv_large": (24, _CONV_LARGE_SPECS),
}


def _get_specs(cfg: MobileNetV4Config) -> tuple[int, list[tuple[int, int, int]]]:
    """Return (stem_channels, block_specs) for the requested variant."""
    if cfg.variant not in _VARIANT_SPECS:
        raise ValueError(
            f"Unknown MobileNetV4 variant '{cfg.variant}'. "
            f"Expected one of: {sorted(_VARIANT_SPECS)}"
        )
    return _VARIANT_SPECS[cfg.variant]


def _build_features(cfg: MobileNetV4Config) -> tuple[nn.Sequential, int]:
    """Return (features, num_out_channels)."""
    stem_ch, specs = _get_specs(cfg)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.ReLU6(inplace=True),
    ]

    in_ch = stem_ch
    for out_ch, stride, expand_ratio in specs:
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

        _, specs = _get_specs(config)
        cumulative = 2  # stem stride=2
        fi: list[FeatureInfo] = []
        for i, (out_ch, s, _) in enumerate(specs):
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
