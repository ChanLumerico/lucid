"""MobileNet v2 configuration (Sandler et al., 2018)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MobileNetV2Config(ModelConfig):
    """Configuration for MobileNet v2.

    ``width_mult`` — uniform channel multiplier; 1.0 = full model.
    ``dropout``    — classifier dropout probability.
    """

    model_type: ClassVar[str] = "mobilenet_v2"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    dropout: float = 0.2
