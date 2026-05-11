"""AlexNet backbone and classifier (Krizhevsky, Sutskever & Hinton, 2012).

Paper: "ImageNet Classification with Deep Convolutional Neural Networks"
Architecture:
    Conv1 : 3→96,  11×11, stride=4, pad=2  → ReLU → LRN → MaxPool 3×3 s2
    Conv2 : 96→256,  5×5,  pad=2            → ReLU → LRN → MaxPool 3×3 s2
    Conv3 : 256→384, 3×3,  pad=1            → ReLU
    Conv4 : 384→384, 3×3,  pad=1            → ReLU
    Conv5 : 384→256, 3×3,  pad=1            → ReLU → MaxPool 3×3 s2
    AdaptiveAvgPool → 6×6
    FC6   : 256*6*6 → 4096                  → ReLU → Dropout
    FC7   : 4096    → 4096                  → ReLU → Dropout
    FC8   : 4096    → num_classes

The original paper split conv filters across two GPUs; the merged single-
stream version (as in standard implementations) is used here.

LRN (Local Response Normalisation) is kept for historical accuracy.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.alexnet._config import AlexNetConfig


def _build_features(cfg: AlexNetConfig) -> nn.Sequential:
    return nn.Sequential(
        # Block 1
        nn.Conv2d(cfg.in_channels, 96, 11, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        nn.MaxPool2d(3, stride=2),
        # Block 2
        nn.Conv2d(96, 256, 5, padding=2),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        nn.MaxPool2d(3, stride=2),
        # Block 3
        nn.Conv2d(256, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # Block 4
        nn.Conv2d(384, 384, 3, padding=1),
        nn.ReLU(inplace=True),
        # Block 5
        nn.Conv2d(384, 256, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2),
    )


# ---------------------------------------------------------------------------
# AlexNet backbone  (task="base")
# ---------------------------------------------------------------------------

class AlexNet(PretrainedModel, BackboneMixin):
    """AlexNet feature extractor — outputs the 5-block conv activations.

    ``forward_features`` returns shape ``(B, 256, 6, 6)`` for 224×224 inputs
    (after the AdaptiveAvgPool2d).
    """

    config_class: ClassVar[type[AlexNetConfig]] = AlexNetConfig
    base_model_prefix: ClassVar[str] = "alexnet"

    def __init__(self, config: AlexNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=96,  reduction=4),
            FeatureInfo(stage=2, num_channels=256, reduction=8),
            FeatureInfo(stage=3, num_channels=384, reduction=16),
            FeatureInfo(stage=4, num_channels=384, reduction=16),
            FeatureInfo(stage=5, num_channels=256, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.features(x))
        return cast(Tensor, self.avgpool(x))

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# AlexNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------

class AlexNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """AlexNet with FC6→FC7→classifier head (4096→4096→num_classes)."""

    config_class: ClassVar[type[AlexNetConfig]] = AlexNetConfig
    base_model_prefix: ClassVar[str] = "alexnet"

    def __init__(self, config: AlexNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
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
