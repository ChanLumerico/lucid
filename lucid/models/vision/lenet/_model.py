"""LeNet-5 backbone and classifier (LeCun et al., 1998).

Original paper: "Gradient-Based Learning Applied to Document Recognition"
Architecture (canonical):
    Input   : 1 × 32 × 32
    C1      : Conv 1→6,  5×5, valid  → 6  × 28 × 28
    S2      : AvgPool 2×2             → 6  × 14 × 14
    C3      : Conv 6→16, 5×5, valid  → 16 × 10 × 10
    S4      : AvgPool 2×2             → 16 × 5  × 5
    C5      : Conv 16→120, 5×5, valid → 120 × 1 × 1   (fully-connected in disguise)
    F6      : Linear 120 → 84
    Output  : Linear 84  → num_classes

Activations in the paper are tanh (squashing functions).  The ``activation``
and ``pooling`` config fields let callers switch to the modern ReLU/MaxPool
convention without changing the topology.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.lenet._config import LeNetConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _act(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    return nn.Tanh()


def _pool(name: str) -> nn.Module:
    if name == "max":
        return nn.MaxPool2d(2, stride=2)
    return nn.AvgPool2d(2, stride=2)


# ---------------------------------------------------------------------------
# Shared feature extractor (C1→S2→C3→S4→C5)
# ---------------------------------------------------------------------------

def _build_features(cfg: LeNetConfig) -> nn.Sequential:
    return nn.Sequential(
        # C1
        nn.Conv2d(cfg.in_channels, 6, 5),
        _act(cfg.activation),
        # S2
        _pool(cfg.pooling),
        # C3
        nn.Conv2d(6, 16, 5),
        _act(cfg.activation),
        # S4
        _pool(cfg.pooling),
        # C5 — acts as a fully-connected conv over the 5×5 feature map
        nn.Conv2d(16, 120, 5),
        _act(cfg.activation),
    )


# ---------------------------------------------------------------------------
# LeNet backbone  (task="base")
# ---------------------------------------------------------------------------

class LeNet(PretrainedModel, BackboneMixin):
    """LeNet-5 feature extractor — outputs C5 activations (120-dim spatial).

    ``forward_features`` returns shape ``(B, 120, 1, 1)`` for 32×32 inputs.
    ``forward`` wraps it in ``BaseModelOutput``.
    """

    config_class: ClassVar[type[LeNetConfig]] = LeNetConfig
    base_model_prefix: ClassVar[str] = "lenet"

    def __init__(self, config: LeNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=6,   reduction=2),
            FeatureInfo(stage=2, num_channels=16,  reduction=4),
            FeatureInfo(stage=3, num_channels=120, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        return cast(Tensor, self.features(x))

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# LeNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------

class LeNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """LeNet-5 with F6 (120→84) and output (84→num_classes) fully-connected layers."""

    config_class: ClassVar[type[LeNetConfig]] = LeNetConfig
    base_model_prefix: ClassVar[str] = "lenet"

    def __init__(self, config: LeNetConfig) -> None:
        super().__init__(config)
        self.features = _build_features(config)
        self.f6 = nn.Linear(120, 84)
        self.act_f6 = _act(config.activation)
        self._build_classifier(84, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.features(x))
        # C5 output is (B, 120, 1, 1) — flatten spatial dims
        x = x.flatten(1)
        x = cast(Tensor, self.act_f6(cast(Tensor, self.f6(x))))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
