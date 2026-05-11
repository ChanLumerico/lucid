"""CoAtNet configuration dataclass (Dai et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class CoAtNetConfig(ModelConfig):
    """Configuration for CoAtNet variants.

    ``variant`` selects the preset; the per-stage block counts and channel
    dimensions below are used directly when constructing the model.

    CoAtNet-0 layout (5 stages after the stem):
      S0/S1/S2 — MBConv stages
      S3/S4    — Transformer stages
    """

    model_type: ClassVar[str] = "coatnet"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "coatnet_0"
    blocks_per_stage: tuple[int, ...] = (2, 2, 6, 14, 2)
    dims: tuple[int, ...] = (96, 192, 384, 768, 768)
    # Number of attention heads for Transformer stages (indexed from S3)
    attn_heads: tuple[int, ...] = (6, 6)
    mbconv_expand: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks_per_stage", tuple(self.blocks_per_stage))
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "attn_heads", tuple(self.attn_heads))
