"""CvT configuration dataclass (Wu et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class CvTConfig(ModelConfig):
    """Configuration for Convolutional Vision Transformer (CvT) variants.

    CvT introduces overlapping convolutional token embedding at each stage
    instead of the standard non-overlapping patch embedding used in plain ViT.

    CvT-13 defaults:
      Stage 1 : embed_stride=4, dim=64,  depth=1,  heads=1
      Stage 2 : embed_stride=2, dim=192, depth=2,  heads=3
      Stage 3 : embed_stride=2, dim=384, depth=10, heads=6
    """

    model_type: ClassVar[str] = "cvt"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "cvt_13"
    # Per-stage configuration (3 stages)
    dims: tuple[int, ...] = (64, 192, 384)
    depths: tuple[int, ...] = (1, 2, 10)
    num_heads: tuple[int, ...] = (1, 3, 6)
    embed_strides: tuple[int, ...] = (4, 2, 2)
    mlp_ratio: float = 4.0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
        object.__setattr__(self, "embed_strides", tuple(self.embed_strides))
