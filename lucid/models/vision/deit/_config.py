"""DeiT configuration (Touvron et al., 2021).

Paper: "Training data-efficient image transformers & distillation through attention"
"""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class DeiTConfig(ModelConfig):
    """Configuration for all DeiT variants.

    DeiT extends ViT with a distillation token: a second learnable token
    prepended alongside the [cls] token.  At inference the predictions from
    both tokens are averaged.

    Canonical variants:
      DeiT-Ti/16: dim=192,  depth=12, heads=3,  patch=16
      DeiT-S/16 : dim=384,  depth=12, heads=6,  patch=16
      DeiT-B/16 : dim=768,  depth=12, heads=12, patch=16
      DeiT-B/32 : dim=768,  depth=12, heads=12, patch=32

    ``mlp_ratio`` — MLP hidden dim relative to ``dim`` (4.0 = 4× expansion).
    ``dropout`` — applied after patch embedding and inside MLP blocks.
    ``attention_dropout`` — dropout on attention weights.
    """

    model_type: ClassVar[str] = "deit"

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    dim: int = 192
    depth: int = 12
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
