"""MnasNet configuration (Tan et al., 2019).

Paper: "MnasNet: Platform-Aware Neural Architecture Search for Mobile"
"""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MnasNetConfig(ModelConfig):
    """Configuration for MnasNet variants.

    Canonical variants:
      MnasNet-0.5 : width_mult=0.5
      MnasNet-1.0 : width_mult=1.0  (baseline MnasNet-A1)
      MnasNet-1.3 : width_mult=1.3

    Channel counts are scaled by ``width_mult`` and rounded to the nearest
    multiple of 8 via ``_make_divisible``.  The first stem conv (3→32) and
    last head conv (320→1280) always use the scaled channel counts.
    """

    model_type: ClassVar[str] = "mnasnet"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    dropout: float = 0.2
