"""LeNet-5 configuration (LeCun et al., 1998)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class LeNetConfig(ModelConfig):
    """Configuration for LeNet-5.

    ``activation`` controls the nonlinearity:
      - ``"tanh"``    — original paper (Gradient-Based Learning, 1998)
      - ``"relu"``    — modern convention

    ``pooling`` controls the sub-sampling layers:
      - ``"avg"``     — original paper (average pooling / sub-sampling)
      - ``"max"``     — modern convention

    ``in_channels`` defaults to 1 (grayscale). Set to 3 for RGB inputs,
    though the canonical use-case is MNIST / single-channel images.
    """

    model_type: ClassVar[str] = "lenet"

    num_classes: int = 10
    in_channels: int = 1
    activation: str = "tanh"  # "tanh" | "relu"
    pooling: str = "avg"  # "avg" | "max"
