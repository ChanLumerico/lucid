"""CrossViT configuration dataclass (Chen et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class CrossViTConfig(ModelConfig):
    """Configuration for CrossViT variants.

    Two-branch ViT with small and large patch sizes. Cross-attention
    synchronises information between branches via the CLS tokens.

    CrossViT-9 defaults:
      depth=3, small_dim=128, large_dim=256
      small_patch=12, large_patch=16
      small_heads=4, large_heads=4
    """

    model_type: ClassVar[str] = "crossvit"

    num_classes: int = 1000
    in_channels: int = 3
    image_size: int = 224
    small_patch: int = 12
    large_patch: int = 16
    small_dim: int = 192
    large_dim: int = 384
    small_heads: int = 3
    large_heads: int = 6
    depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
