"""SqueezeNet backbone and classifier (Iandola et al., 2016).

Paper: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
       less than 0.5MB model size"

Fire module:
  squeeze: Conv2d(in_ch, s1x1, 1) → ReLU
  expand_1x1: Conv2d(s1x1, e1x1, 1) → ReLU
  expand_3x3: Conv2d(s1x1, e3x3, 3, padding=1) → ReLU
  output: cat([expand_1x1, expand_3x3], dim=1)  → (e1x1 + e3x3) channels

SqueezeNet 1.0 architecture:
  Conv2d(3, 96, 7, stride=2) → ReLU → MaxPool(3, stride=2, ceil_mode=True)
  Fire(96, 16, 64, 64) → Fire(128, 16, 64, 64) → Fire(128, 32, 128, 128)
  MaxPool(3, stride=2, ceil_mode=True)
  Fire(256, 32, 128, 128) → Fire(256, 48, 192, 192) → Fire(384, 48, 192, 192)
  Fire(384, 64, 256, 256) → MaxPool(3, stride=2, ceil_mode=True)
  Fire(512, 64, 256, 256)
  [classifier: Dropout → Conv2d(512, num_classes, 1) → ReLU → AdaptiveAvgPool]

SqueezeNet 1.1 architecture:
  Conv2d(3, 64, 3, stride=2) → ReLU → MaxPool(3, stride=2, ceil_mode=True)
  Fire(64, 16, 64, 64) → Fire(128, 16, 64, 64)
  MaxPool(3, stride=2, ceil_mode=True)
  Fire(128, 32, 128, 128) → Fire(256, 32, 128, 128)
  MaxPool(3, stride=2, ceil_mode=True)
  Fire(256, 48, 192, 192) → Fire(384, 48, 192, 192) → Fire(384, 64, 256, 256)
  Fire(512, 64, 256, 256)
  [classifier: Dropout → Conv2d(512, num_classes, 1) → ReLU → AdaptiveAvgPool]
"""

import lucid
from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.squeezenet._config import SqueezeNetConfig

# ---------------------------------------------------------------------------
# Fire module
# ---------------------------------------------------------------------------


class _FireModule(nn.Module):
    """Squeeze-then-parallel-expand Fire module."""

    def __init__(
        self,
        in_ch: int,
        s1x1: int,
        e1x1: int,
        e3x3: int,
    ) -> None:
        super().__init__()
        self.squeeze = nn.Conv2d(in_ch, s1x1, 1)
        self.expand_1x1 = nn.Conv2d(s1x1, e1x1, 1)
        self.expand_3x3 = nn.Conv2d(s1x1, e3x3, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = F.relu(cast(Tensor, self.squeeze(x)))
        e1 = F.relu(cast(Tensor, self.expand_1x1(x)))
        e3 = F.relu(cast(Tensor, self.expand_3x3(x)))
        return lucid.cat([e1, e3], dim=1)


# ---------------------------------------------------------------------------
# Architecture builders
# ---------------------------------------------------------------------------


def _build_features_1_0(in_channels: int) -> tuple[nn.Sequential, int]:
    """Build SqueezeNet 1.0 feature layers (everything before the classifier)."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, 96, 7, stride=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
        _FireModule(96,  16, 64,  64),
        _FireModule(128, 16, 64,  64),
        _FireModule(128, 32, 128, 128),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
        _FireModule(256, 32, 128, 128),
        _FireModule(256, 48, 192, 192),
        _FireModule(384, 48, 192, 192),
        _FireModule(384, 64, 256, 256),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
        _FireModule(512, 64, 256, 256),
    ]
    return nn.Sequential(*layers), 512


def _build_features_1_1(in_channels: int) -> tuple[nn.Sequential, int]:
    """Build SqueezeNet 1.1 feature layers (everything before the classifier)."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_channels, 64, 3, stride=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
        _FireModule(64,  16, 64,  64),
        _FireModule(128, 16, 64,  64),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
        _FireModule(128, 32, 128, 128),
        _FireModule(256, 32, 128, 128),
        nn.MaxPool2d(3, stride=2, ceil_mode=True),
        _FireModule(256, 48, 192, 192),
        _FireModule(384, 48, 192, 192),
        _FireModule(384, 64, 256, 256),
        _FireModule(512, 64, 256, 256),
    ]
    return nn.Sequential(*layers), 512


def _build_features(
    cfg: SqueezeNetConfig,
) -> tuple[nn.Sequential, int]:
    if cfg.version == "1_0":
        return _build_features_1_0(cfg.in_channels)
    return _build_features_1_1(cfg.in_channels)


# ---------------------------------------------------------------------------
# SqueezeNet backbone (task="base")
# ---------------------------------------------------------------------------


class SqueezeNet(PretrainedModel, BackboneMixin):
    """SqueezeNet feature extractor — outputs pooled (B, 512, 1, 1).

    The backbone returns the global-averaged features after the last Fire
    module (before the final classification conv).
    """

    config_class: ClassVar[type[SqueezeNetConfig]] = SqueezeNetConfig
    base_model_prefix: ClassVar[str] = "squeezenet"

    def __init__(self, config: SqueezeNetConfig) -> None:
        super().__init__(config)
        features, num_features = _build_features(config)
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features: int = num_features

        # SqueezeNet has a single-scale output: 512 channels, stride-32
        self._feature_info: list[FeatureInfo] = [
            FeatureInfo(stage=1, num_channels=num_features, reduction=32),
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
# SqueezeNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class SqueezeNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """SqueezeNet with final Conv2d → ReLU → AdaptiveAvgPool classifier.

    The classification head is a Conv2d(512, num_classes, 1) followed by
    global average pooling — no separate Linear layer, matching the original
    paper's "convolutional classifier" design.  We expose this through the
    standard ``self.classifier`` attribute (a Conv2d) for API consistency.
    """

    config_class: ClassVar[type[SqueezeNetConfig]] = SqueezeNetConfig
    base_model_prefix: ClassVar[str] = "squeezenet"

    def __init__(self, config: SqueezeNetConfig) -> None:
        super().__init__(config)
        features, _ = _build_features(config)
        self.features = features
        self.drop = nn.Dropout(p=config.dropout)
        # Convolutional classifier (paper design — no Linear)
        self.classifier = nn.Conv2d(512, config.num_classes, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.features(x))
        x = cast(Tensor, self.drop(x))
        x = F.relu(cast(Tensor, self.classifier(x)))
        x = cast(Tensor, self.avgpool(x))
        # (B, num_classes, 1, 1) → (B, num_classes)
        logits = x.flatten(1)

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
