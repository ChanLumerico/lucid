"""VGG backbone and classifier (Simonyan & Zisserman, 2014).

Paper: "Very Deep Convolutional Networks for Large-Scale Image Recognition"
All variants share the same macro structure:
  5 blocks of (N × Conv3×3 → ReLU [→ BN]) → MaxPool
  → AdaptiveAvgPool(7×7)
  → FC(512*7*7, 4096) → ReLU → Dropout
  → FC(4096, 4096)   → ReLU → Dropout
  → FC(4096, num_classes)

Channel widths: [64, 128, 256, 512, 512] — fixed across all VGG variants;
only the per-block conv count changes.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.vgg._config import VGGConfig

_CHANNELS = (64, 128, 256, 512, 512)


def _make_block(
    in_ch: int, out_ch: int, num_convs: int, batch_norm: bool
) -> list[nn.Module]:
    layers: list[nn.Module] = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        in_ch = out_ch
    layers.append(nn.MaxPool2d(2, stride=2))
    return layers


def _build_features(cfg: VGGConfig) -> nn.Sequential:
    layers: list[nn.Module] = []
    in_ch = cfg.in_channels
    for out_ch, n in zip(_CHANNELS, cfg.arch):
        layers.extend(_make_block(in_ch, out_ch, n, cfg.batch_norm))
        in_ch = out_ch
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# VGG backbone  (task="base")
# ---------------------------------------------------------------------------


class VGG(PretrainedModel, BackboneMixin):
    """VGG feature extractor — outputs 5-block conv activations.

    ``forward_features`` returns shape ``(B, 512, 7, 7)`` for 224×224 inputs
    after AdaptiveAvgPool2d.
    """

    config_class: ClassVar[type[VGGConfig]] = VGGConfig
    base_model_prefix: ClassVar[str] = "vgg"

    def __init__(self, config: VGGConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self._feature_info = [
            FeatureInfo(stage=i + 1, num_channels=ch, reduction=2 ** (i + 1))
            for i, ch in enumerate(_CHANNELS)
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
# VGG for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class VGGForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """VGG with two 4096-dim FC layers + final classifier."""

    config_class: ClassVar[type[VGGConfig]] = VGGConfig
    base_model_prefix: ClassVar[str] = "vgg"

    def __init__(self, config: VGGConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.fc6 = nn.Linear(512 * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.drop6 = nn.Dropout(p=config.dropout)
        self.drop7 = nn.Dropout(p=config.dropout)
        self._build_classifier(4096, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.features(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        x = cast(Tensor, self.drop6(F.relu(cast(Tensor, self.fc6(x)))))
        x = cast(Tensor, self.drop7(F.relu(cast(Tensor, self.fc7(x)))))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
