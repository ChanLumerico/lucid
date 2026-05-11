"""Inception-ResNet v2 configuration (Szegedy et al., 2016)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class InceptionResNetConfig(ModelConfig):
    """Configuration for Inception-ResNet v2.

    Uses the same stem as Inception-v4, then Inception-A/B/C blocks with
    residual connections scaled by ``scale``.

    ``scale`` — residual branch scale factor (0.1–0.3, paper default 0.1).
    ``dropout`` — head dropout rate (0.2 in the paper).
    """

    model_type: ClassVar[str] = "inception_resnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.2
    scale: float = 0.1
