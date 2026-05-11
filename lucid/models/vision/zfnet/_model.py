"""ZFNet backbone and classification head (Zeiler & Fergus, 2013).

Paper: "Visualizing and Understanding Convolutional Networks"
Architecture differs from AlexNet in the first two conv layers:
    Conv1 : 3→96,  7×7, stride=2, pad=1  → ReLU → LRN → MaxPool 3×3 s2
    Conv2 : 96→256, 5×5, stride=2, pad=0 → ReLU → LRN → MaxPool 3×3 s2
    Conv3 : 256→384, 3×3, pad=1          → ReLU
    Conv4 : 384→384, 3×3, pad=1          → ReLU
    Conv5 : 384→256, 3×3, pad=1          → ReLU → MaxPool 3×3 s2
    AdaptiveAvgPool → 6×6
    FC6   : 256*6*6 → 4096               → ReLU → Dropout
    FC7   : 4096    → 4096               → ReLU → Dropout
    FC8   : 4096    → num_classes
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.zfnet._config import ZFNetConfig


def _build_features(cfg: ZFNetConfig) -> nn.Sequential:
    return nn.Sequential(
        # Block 1 — 7×7 stride=2 (key ZFNet modification)
        nn.Conv2d(cfg.in_channels, 96, 7, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=2.0),
        nn.MaxPool2d(3, stride=2),
        # Block 2 — 5×5 stride=2 (key ZFNet modification)
        nn.Conv2d(96, 256, 5, stride=2),
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
# ZFNet backbone  (task="base")
# ---------------------------------------------------------------------------


class ZFNet(PretrainedModel, BackboneMixin):
    """ZFNet feature extractor — outputs the 5-block conv activations.

    ``forward_features`` returns shape ``(B, 256, 6, 6)`` for 224×224 inputs
    (after the AdaptiveAvgPool2d).
    """

    config_class: ClassVar[type[ZFNetConfig]] = ZFNetConfig
    base_model_prefix: ClassVar[str] = "zfnet"

    def __init__(self, config: ZFNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=96, reduction=4),
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

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# ZFNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ZFNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """ZFNet with FC6→FC7→classifier head (4096→4096→num_classes)."""

    config_class: ClassVar[type[ZFNetConfig]] = ZFNetConfig
    base_model_prefix: ClassVar[str] = "zfnet"

    def __init__(self, config: ZFNetConfig) -> None:
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
