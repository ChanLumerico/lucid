"""Swin Transformer V2 configuration (Liu et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SwinV2Config(ModelConfig):
    """Configuration for Swin Transformer V2 variants (tiny/small/base/large).

    Key differences from V1:
      - ``image_size`` default is 256 (V1 was 224).
      - ``window_size`` default is 8 (V1 was 7).
      - Scaled cosine self-attention replaces dot-product attention.
      - Log-spaced Continuous Position Bias (CPB) MLP replaces the learned table.
      - Post-normalization (LayerNorm after attention/MLP, not before).

    Canonical variants:
      tiny  : embed_dim=96,  depths=(2,2,6,2),   num_heads=(3,6,12,24)
      small : embed_dim=96,  depths=(2,2,18,2),  num_heads=(3,6,12,24)
      base  : embed_dim=128, depths=(2,2,18,2),  num_heads=(4,8,16,32)
      large : embed_dim=192, depths=(2,2,18,2),  num_heads=(6,12,24,48)
    """

    model_type: ClassVar[str] = "swin_v2"

    image_size: int = 256
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
