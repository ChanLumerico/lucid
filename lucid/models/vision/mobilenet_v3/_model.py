"""MobileNet v3 backbone and classifier (Howard et al., 2019).

Paper: "Searching for MobileNetV3"

Key innovations:
  - Hard-swish and Hard-sigmoid activations for mobile efficiency.
  - Squeeze-and-Excitation (SE) blocks with Hard-sigmoid gating.
  - NAS-designed architecture for Large and Small variants.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.mobilenet_v3._config import MobileNetV3Config


def _make_divisible(v: float, divisor: int = 8) -> int:
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation block (Hard-sigmoid gating)
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    def __init__(self, in_channels: int, se_channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, se_channels, 1)
        self.fc2 = nn.Conv2d(se_channels, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        scale = cast(Tensor, self.pool(x))
        scale = F.relu(cast(Tensor, self.fc1(scale)), inplace=True)
        scale = F.hardsigmoid(cast(Tensor, self.fc2(scale)))
        return x * scale


# ---------------------------------------------------------------------------
# Bottleneck block (InvertedResidual + optional SE + configurable activation)
# ---------------------------------------------------------------------------


class _InvertedResidual(nn.Module):
    """MobileNet v3 bottleneck with optional SE and activation choice."""

    def __init__(
        self,
        in_ch: int,
        exp_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int,
        use_se: bool,
        use_hs: bool,
    ) -> None:
        super().__init__()
        self._use_residual = stride == 1 and in_ch == out_ch

        act: nn.Module = nn.Hardswish() if use_hs else nn.ReLU(inplace=True)
        layers: list[nn.Module] = []

        if exp_ch != in_ch:
            layers += [
                nn.Conv2d(in_ch, exp_ch, 1, bias=False),
                nn.BatchNorm2d(exp_ch),
                act,
            ]
        pad = (kernel_size - 1) // 2
        act2: nn.Module = nn.Hardswish() if use_hs else nn.ReLU(inplace=True)
        layers += [
            nn.Conv2d(
                exp_ch,
                exp_ch,
                kernel_size,
                stride=stride,
                padding=pad,
                groups=exp_ch,
                bias=False,
            ),
            nn.BatchNorm2d(exp_ch),
            act2,
        ]

        if use_se:
            se_ch = _make_divisible(exp_ch * 0.25)
            layers.append(_SEBlock(exp_ch, se_ch))

        layers += [
            nn.Conv2d(exp_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = cast(Tensor, self.block(x))
        if self._use_residual:
            out = out + x
        return out


# ---------------------------------------------------------------------------
# Architecture specs
# (in_ch, exp_ch, out_ch, kernel_size, use_se, use_hs, stride)
# ---------------------------------------------------------------------------

_LARGE_SPECS: list[tuple[int, int, int, int, bool, bool, int]] = [
    (16, 16, 16, 3, False, False, 1),
    (16, 64, 24, 3, False, False, 2),
    (24, 72, 24, 3, False, False, 1),
    (24, 72, 40, 5, True, False, 2),
    (40, 120, 40, 5, True, False, 1),
    (40, 120, 40, 5, True, False, 1),
    (40, 240, 80, 3, False, True, 2),
    (80, 200, 80, 3, False, True, 1),
    (80, 184, 80, 3, False, True, 1),
    (80, 184, 80, 3, False, True, 1),
    (80, 480, 112, 3, True, True, 1),
    (112, 672, 112, 3, True, True, 1),
    (112, 672, 160, 5, True, True, 2),
    (160, 960, 160, 5, True, True, 1),
    (160, 960, 160, 5, True, True, 1),
]

_SMALL_SPECS: list[tuple[int, int, int, int, bool, bool, int]] = [
    (16, 16, 16, 3, True, False, 2),
    (16, 72, 24, 3, False, False, 2),
    (24, 88, 24, 3, False, False, 1),
    (24, 96, 40, 5, True, True, 2),
    (40, 240, 40, 5, True, True, 1),
    (40, 240, 40, 5, True, True, 1),
    (40, 120, 48, 5, True, True, 1),
    (48, 144, 48, 5, True, True, 1),
    (48, 288, 96, 5, True, True, 2),
    (96, 576, 96, 5, True, True, 1),
    (96, 576, 96, 5, True, True, 1),
]


def _apply_width(
    specs: list[tuple[int, int, int, int, bool, bool, int]],
    w: float,
) -> list[tuple[int, int, int, int, bool, bool, int]]:
    """Scale channel counts by width_mult."""
    out: list[tuple[int, int, int, int, bool, bool, int]] = []
    for in_ch, exp_ch, o_ch, k, se, hs, s in specs:
        out.append(
            (
                _make_divisible(in_ch * w),
                _make_divisible(exp_ch * w),
                _make_divisible(o_ch * w),
                k,
                se,
                hs,
                s,
            )
        )
    return out


def _build_features(cfg: MobileNetV3Config) -> tuple[nn.Sequential, int, int]:
    """Return (features, last_block_ch, head_out_ch)."""
    w = cfg.width_mult
    specs = _LARGE_SPECS if cfg.variant == "large" else _SMALL_SPECS

    stem_ch = _make_divisible(16 * w)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.Hardswish(),
    ]

    scaled = _apply_width(specs, w)
    in_ch = stem_ch
    for spec_in, exp_ch, out_ch, k, se, hs, stride in scaled:
        # Clamp first in_ch to stem_ch; subsequent blocks chain naturally
        layers.append(_InvertedResidual(in_ch, exp_ch, out_ch, k, stride, se, hs))
        in_ch = out_ch

    # Head conv
    last_block_ch = in_ch
    penultimate_ch = (
        _make_divisible(960 * w) if cfg.variant == "large" else _make_divisible(576 * w)
    )
    layers += [
        nn.Conv2d(last_block_ch, penultimate_ch, 1, bias=False),
        nn.BatchNorm2d(penultimate_ch),
        nn.Hardswish(),
    ]
    # Large → 1280, Small → 1024 (paper Table 2 / torchvision)
    head_ch = (
        _make_divisible(1280 * w)
        if cfg.variant == "large"
        else _make_divisible(1024 * w)
    )
    return nn.Sequential(*layers), penultimate_ch, head_ch


def _build_classifier_head(
    penultimate_ch: int,
    head_ch: int,
    num_classes: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Conv2d(penultimate_ch, head_ch, 1),
        nn.Hardswish(),
        nn.Flatten(),
        nn.Dropout(p=dropout),
        nn.Linear(head_ch, num_classes),
    )


# ---------------------------------------------------------------------------
# MobileNet v3 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV3(PretrainedModel, BackboneMixin):
    """MobileNet v3 feature extractor (Large or Small)."""

    config_class: ClassVar[type[MobileNetV3Config]] = MobileNetV3Config
    base_model_prefix: ClassVar[str] = "mobilenet_v3"

    def __init__(self, config: MobileNetV3Config) -> None:
        super().__init__(config)
        features, num_features, _ = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult
        specs = _LARGE_SPECS if config.variant == "large" else _SMALL_SPECS
        scaled = _apply_width(specs, w)

        cumulative = 2  # stem stride=2
        fi: list[FeatureInfo] = []
        for i, (_, _, o_ch, _, _, _, s) in enumerate(scaled):
            cumulative *= s
            fi.append(FeatureInfo(stage=i + 1, num_channels=o_ch, reduction=cumulative))
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
# MobileNet v3 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV3ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """MobileNet v3 with inverted-classifier head."""

    config_class: ClassVar[type[MobileNetV3Config]] = MobileNetV3Config
    base_model_prefix: ClassVar[str] = "mobilenet_v3"

    def __init__(self, config: MobileNetV3Config) -> None:
        super().__init__(config)
        features, penultimate_ch, head_ch = _build_features(config)
        self.features = features
        self.classifier = _build_classifier_head(
            penultimate_ch, head_ch, config.num_classes, config.dropout
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.features(x))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
