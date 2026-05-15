"""MobileNet v1 backbone and classifier (Howard et al., 2017).

Paper: "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"

Key idea: replace standard Conv2d with *depthwise separable convolutions* —
  Depthwise  : one filter per input channel (groups=in_channels)
  Pointwise  : 1×1 Conv to project to the desired output channels

This reduces computation from M·N·Dk²·Df² to M·Dk²·Df² + M·N·Df²,
roughly a 8–9× reduction for 3×3 convolutions.

Architecture (width_mult=1.0):
  Conv   : 3→32, 3×3, s2
  DW+PW  : 32→64, s1
  DW+PW  : 64→128, s2
  DW+PW  : 128→128, s1
  DW+PW  : 128→256, s2
  DW+PW  : 256→256, s1
  DW+PW  : 256→512, s2
  DW+PW × 5 : 512→512, s1
  DW+PW  : 512→1024, s2
  DW+PW  : 1024→1024, s1
  AvgPool(7×7) → FC(1024, num_classes)
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import make_divisible as _make_divisible
from lucid.models.vision.mobilenet._config import MobileNetV1Config


def _dw_pw(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    """Depthwise + pointwise block with BN and ReLU."""
    return nn.Sequential(
        # Depthwise
        nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
        nn.BatchNorm2d(in_ch),
        nn.ReLU(inplace=True),
        # Pointwise
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# (out_channels, stride) specs for the 13 DW+PW layers (at width_mult=1.0)
_DW_PW_SPECS: list[tuple[int, int]] = [
    (64, 1),
    (128, 2),
    (128, 1),
    (256, 2),
    (256, 1),
    (512, 2),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (512, 1),
    (1024, 2),
    (1024, 1),
]


def _build_features(cfg: MobileNetV1Config) -> tuple[nn.Sequential, int]:
    """Build the full feature extractor. Returns (Sequential, num_out_channels)."""
    w = cfg.width_mult

    def _ch(c: int) -> int:
        return _make_divisible(c * w)

    first_ch = _ch(32)
    layers: list[nn.Module] = [
        nn.Conv2d(cfg.in_channels, first_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(first_ch),
        nn.ReLU(inplace=True),
    ]
    in_ch = first_ch
    for out_c, stride in _DW_PW_SPECS:
        out_ch = _ch(out_c)
        layers.append(_dw_pw(in_ch, out_ch, stride))
        in_ch = out_ch

    return nn.Sequential(*layers), in_ch


# ---------------------------------------------------------------------------
# MobileNet v1 backbone  (task="base")
# ---------------------------------------------------------------------------


class MobileNetV1(PretrainedModel, BackboneMixin):
    """MobileNet v1 feature extractor — 1024-ch spatial map (1×1 after AvgPool)."""

    config_class: ClassVar[type[MobileNetV1Config]] = MobileNetV1Config
    base_model_prefix: ClassVar[str] = "mobilenet_v1"

    def __init__(self, config: MobileNetV1Config) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        w = config.width_mult

        def _ch(c: int) -> int:
            return _make_divisible(c * w)

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=_ch(64), reduction=2),
            FeatureInfo(stage=2, num_channels=_ch(128), reduction=4),
            FeatureInfo(stage=3, num_channels=_ch(256), reduction=8),
            FeatureInfo(stage=4, num_channels=_ch(512), reduction=16),
            FeatureInfo(stage=5, num_channels=_ch(1024), reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.features(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# MobileNet v1 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class MobileNetV1ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """MobileNet v1 with AvgPool + Dropout + FC classifier."""

    config_class: ClassVar[type[MobileNetV1Config]] = MobileNetV1Config
    base_model_prefix: ClassVar[str] = "mobilenet_v1"

    def __init__(self, config: MobileNetV1Config) -> None:
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
