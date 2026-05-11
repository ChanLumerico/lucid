"""Inception v4 configuration (Szegedy et al., 2016)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class InceptionV4Config(ModelConfig):
    """Configuration for Inception v4.

    Cleaner architecture than v3 with dedicated Inception-A/B/C blocks and
    Reduction-A/B modules.  Designed for 299×299 inputs.

    ``dropout`` — head dropout rate (0.2 in the paper).
    """

    model_type: ClassVar[str] = "inception_v4"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.2
