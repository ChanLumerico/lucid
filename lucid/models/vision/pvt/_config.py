"""PVT (Pyramid Vision Transformer) configuration (Wang et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class PVTConfig(ModelConfig):
    """Configuration for PVT variants.

    Canonical variants:
      PVT-Tiny: embed_dims=(64,128,320,512), depths=(2,2,2,2),
                num_heads=(1,2,5,8), sr_ratios=(8,4,2,1)

    ``embed_dims`` — output channels per stage.
    ``depths``     — transformer blocks per stage.
    ``num_heads``  — attention heads per stage.
    ``sr_ratios``  — spatial reduction ratio per stage (1 = no reduction).
    ``mlp_ratio``  — MLP hidden expansion (8.0 in the paper).
    ``variant``    — informational variant label.
    """

    model_type: ClassVar[str] = "pvt"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "pvt_tiny"
    embed_dims: tuple[int, ...] = (64, 128, 320, 512)
    depths: tuple[int, ...] = (2, 2, 2, 2)
    num_heads: tuple[int, ...] = (1, 2, 5, 8)
    sr_ratios: tuple[int, ...] = (8, 4, 2, 1)
    mlp_ratio: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "embed_dims", tuple(self.embed_dims))
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
        object.__setattr__(self, "sr_ratios", tuple(self.sr_ratios))
