"""MobileNet v1 configuration (Howard et al., 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MobileNetV1Config(ModelConfig):
    """Configuration for MobileNet v1.

    ``width_mult`` — uniform channel multiplier (α in the paper).
      1.0 → full model; 0.75 / 0.5 / 0.25 → slimmer variants.

    ``resolution_mult`` — spatial resolution multiplier (ρ).
      Typically applied to the input, not the architecture itself.
      Kept here for documentation parity; not enforced inside the model.

    ``dropout`` — classifier dropout (0.001 in the original paper).
    """

    model_type: ClassVar[str] = "mobilenet_v1"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    dropout: float = 0.001
