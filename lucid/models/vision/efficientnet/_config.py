"""EfficientNet configuration (Tan & Le, 2019)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class EfficientNetConfig(ModelConfig):
    """Configuration for EfficientNet B0–B7.

    Compound scaling coefficients:
      B0: width=1.0, depth=1.0, res=224, dropout=0.2
      B1: width=1.0, depth=1.1, res=240, dropout=0.2
      B2: width=1.1, depth=1.2, res=260, dropout=0.3
      B3: width=1.2, depth=1.4, res=300, dropout=0.3
      B4: width=1.4, depth=1.8, res=380, dropout=0.4
      B5: width=1.6, depth=2.2, res=456, dropout=0.4
      B6: width=1.8, depth=2.6, res=528, dropout=0.5
      B7: width=2.0, depth=3.1, res=600, dropout=0.5

    ``drop_connect_rate`` — stochastic depth rate applied linearly across blocks.
    ``se_ratio`` — squeeze-and-excitation reduction ratio (0.25 in the paper).
    """

    model_type: ClassVar[str] = "efficientnet"

    num_classes: int = 1000
    in_channels: int = 3
    width_mult: float = 1.0
    depth_mult: float = 1.0
    dropout: float = 0.2
    drop_connect_rate: float = 0.2
    se_ratio: float = 0.25
