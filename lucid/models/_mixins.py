"""Composable mixins for model families.

Tier-2 reusable behaviour — added only when ≥3 families would otherwise
duplicate the same logic.  Mixins carry no state (no ``__init__`` of
their own); they expose methods that operate on attributes the host class
is contracted to provide.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import lucid.nn as nn

if TYPE_CHECKING:
    from lucid._tensor.tensor import Tensor


@dataclass(frozen=True)
class FeatureInfo:
    """Output spec for one stage of a backbone (timm-compatible)."""

    stage: int
    num_channels: int
    reduction: int  # spatial down-sampling factor vs the input


class BackboneMixin(ABC):
    """Marker mixin: a model that can serve as a feature extractor.

    Subclasses must implement :meth:`forward_features` returning the
    deepest stage's feature map and :attr:`feature_info` enumerating
    every emitted stage's channel / stride spec.
    """

    @abstractmethod
    def forward_features(self, x: Tensor) -> Tensor: ...

    @property
    @abstractmethod
    def feature_info(self) -> list[FeatureInfo]: ...


class ClassificationHeadMixin:
    """Standard ``classifier`` Linear head + transfer-learning hook.

    Subclasses must call :meth:`_build_classifier` in their ``__init__``
    *after* ``super().__init__(config)`` to install the head.  Use
    :meth:`reset_classifier` to swap ``num_classes`` post hoc.
    """

    classifier: nn.Module

    def _build_classifier(
        self,
        in_features: int,
        num_classes: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        if dropout > 0.0:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        else:
            self.classifier = nn.Linear(in_features, num_classes)

    def reset_classifier(self, num_classes: int) -> None:
        """Replace the final Linear with a freshly initialised one."""
        if isinstance(self.classifier, nn.Linear):
            in_features = int(self.classifier.in_features)
            self.classifier = nn.Linear(in_features, num_classes)
            return
        if isinstance(self.classifier, nn.Sequential):
            for i in range(len(self.classifier) - 1, -1, -1):
                m = self.classifier[i]
                if isinstance(m, nn.Linear):
                    in_features = int(m.in_features)
                    self.classifier[i] = nn.Linear(in_features, num_classes)
                    return
            raise RuntimeError(
                "reset_classifier: no Linear layer found in classifier Sequential"
            )
        raise NotImplementedError(
            f"reset_classifier not implemented for " f"{type(self.classifier).__name__}"
        )
