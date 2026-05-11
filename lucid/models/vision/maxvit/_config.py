"""MaxViT configuration (Tu et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MaxViTConfig(ModelConfig):
    """Configuration for MaxViT variants.

    Canonical variants:
      MaxViT-T: depths=(2,2,5,2), dims=(64,128,256,512), num_heads=32
      MaxViT-S: depths=(2,2,5,2), dims=(96,192,384,768), num_heads=32
      MaxViT-B: depths=(2,6,14,2), dims=(96,192,384,768), num_heads=32

    Each MaxViT block = MBConv + block-attention (local window) + grid-attention.
    ``window_size`` — local window size (8 in the paper).
    ``num_heads``   — attention heads shared across block and grid attention.
    ``mlp_ratio``   — MLP expansion ratio inside attention blocks.
    """

    model_type: ClassVar[str] = "maxvit"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (2, 2, 5, 2)
    dims: tuple[int, ...] = (64, 128, 256, 512)
    window_size: int = 7
    num_heads: int = 32
    mlp_ratio: float = 4.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "dims", tuple(self.dims))
