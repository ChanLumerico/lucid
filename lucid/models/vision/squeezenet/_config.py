"""SqueezeNet configuration (Iandola et al., 2016)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SqueezeNetConfig(ModelConfig):
    """Configuration for SqueezeNet 1.0 and 1.1.

    ``version`` — "1_0" (original) or "1_1" (faster, ~2.4× less compute).
    The two versions differ in the stem conv size and the placement of
    MaxPool layers between Fire modules.
    """

    model_type: ClassVar[str] = "squeezenet"

    version: str = "1_1"  # "1_0" or "1_1"
    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
