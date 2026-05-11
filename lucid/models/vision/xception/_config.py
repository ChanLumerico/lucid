"""Xception configuration (Chollet, 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class XceptionConfig(ModelConfig):
    """Configuration for Xception (Extreme Inception).

    Replaces all Inception modules with depthwise separable convolutions.
    Designed for 299×299 inputs.

    ``dropout`` — head dropout rate (0.5 in the paper).
    """

    model_type: ClassVar[str] = "xception"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
