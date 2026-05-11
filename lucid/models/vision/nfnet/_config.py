"""NFNet configuration (Brock et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class NFNetConfig(ModelConfig):
    """Configuration for Normalizer-Free Networks (NFNet-F0 through F3).

    NFNet removes BatchNorm entirely, relying on:
      - Scaled Weight Standardization (ScaledStdConv2d) instead of BN
      - Squeeze-Excitation gating on residual branches
      - Stochastic depth for regularisation
      - alpha/beta variance bookkeeping across blocks

    Canonical variants (from the paper):
      NFNet-F0 : widths=(256,512,1536,1536), depths=(1,2,6,3),   groups=128
      NFNet-F1 : widths=(256,512,1536,1536), depths=(2,4,12,6),  groups=128
      NFNet-F2 : widths=(256,512,1536,1536), depths=(3,6,18,9),  groups=128
      NFNet-F3 : widths=(256,512,1536,1536), depths=(4,8,24,12), groups=128

    ``alpha``       — per-block residual scale (0.2); controls expected variance growth.
    ``se_ratio``    — squeeze ratio for SE block (0.5 × in_ch).
    ``group_size``  — depthwise-grouped conv group size (128); mid_ch must be divisible.
    ``dropout``     — classifier dropout (0.2 for F0).
    ``stoch_depth`` — stochastic depth drop rate (uniform across blocks).
    """

    model_type: ClassVar[str] = "nfnet"

    num_classes: int = 1000
    in_channels: int = 3
    widths: tuple[int, ...] = (256, 512, 1536, 1536)
    depths: tuple[int, ...] = (1, 2, 6, 3)
    group_size: int = 128
    alpha: float = 0.2
    se_ratio: float = 0.5
    dropout: float = 0.2
    stoch_depth: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "widths", tuple(self.widths))
        object.__setattr__(self, "depths", tuple(self.depths))
