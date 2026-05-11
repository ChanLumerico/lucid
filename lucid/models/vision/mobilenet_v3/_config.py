"""MobileNet v3 configuration (Howard et al., 2019)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MobileNetV3Config(ModelConfig):
    """Configuration for MobileNet v3.

    ``variant``    — "large" or "small".
    ``width_mult`` — uniform channel multiplier.
    ``dropout``    — classifier dropout probability.
    """

    model_type: ClassVar[str] = "mobilenet_v3"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "large"
    width_mult: float = 1.0
    dropout: float = 0.2
