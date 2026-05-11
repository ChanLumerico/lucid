"""MobileNet v4 configuration (Qin et al., 2024)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MobileNetV4Config(ModelConfig):
    """Configuration for MobileNet v4.

    ``variant`` — one of "conv_small", "conv_medium", "conv_large".
    """

    model_type: ClassVar[str] = "mobilenet_v4"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "conv_small"
    dropout: float = 0.2
