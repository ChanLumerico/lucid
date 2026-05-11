"""Vision Transformer (ViT) configuration (Dosovitskiy et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ViTConfig(ModelConfig):
    """Configuration for all ViT variants.

    Canonical variants:
      ViT-B/16 : dim=768,  depth=12, heads=12, patch=16
      ViT-B/32 : dim=768,  depth=12, heads=12, patch=32
      ViT-L/16 : dim=1024, depth=24, heads=16, patch=16
      ViT-L/32 : dim=1024, depth=24, heads=16, patch=32
      ViT-H/14 : dim=1280, depth=32, heads=16, patch=14

    ``mlp_ratio`` — MLP hidden dim relative to ``dim`` (4.0 = 4× expansion).
    ``dropout`` — applied after patch embedding and inside MLP blocks.
    ``attention_dropout`` — dropout on attention weights.
    """

    model_type: ClassVar[str] = "vit"

    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 1000
    in_channels: int = 3
    dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
