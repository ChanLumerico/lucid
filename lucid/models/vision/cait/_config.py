"""CaiT configuration (Touvron et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class CaiTConfig(ModelConfig):
    """Configuration for Class Attention in Image Transformers (CaiT).

    CaiT extends ViT with two innovations:
      - LayerScale: per-channel learnable scaling (γ) on residual branches.
      - Class Attention (CA) layers: after the self-attention blocks, 2 CA
        layers process only the class token (cls queries all tokens; patches
        are frozen in those layers).

    Canonical variants (from Touvron et al., 2021, Table 1):
      xxsmall_24  : dim=192,  depth=24, heads=4,  ls_init=1e-5
      xxsmall_36  : dim=192,  depth=36, heads=4,  ls_init=1e-5
      xsmall_24   : dim=288,  depth=24, heads=6,  ls_init=1e-5
      small_24    : dim=384,  depth=24, heads=8,  ls_init=1e-5
      small_36    : dim=384,  depth=36, heads=8,  ls_init=1e-5
      medium_36   : dim=768,  depth=36, heads=16, ls_init=1e-6
      medium_48   : dim=768,  depth=48, heads=16, ls_init=1e-6

    ``depth``            — number of self-attention blocks (patch tokens only).
    ``class_depth``      — number of class-attention blocks (always 2 in the paper).
    ``layer_scale_init`` — initial value for the per-channel LayerScale γ.
    ``attention_dropout``— dropout applied to attention weights.
    """

    model_type: ClassVar[str] = "cait"

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    dim: int = 192
    depth: int = 24
    num_heads: int = 4
    mlp_ratio: float = 4.0
    class_depth: int = 2
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_scale_init: float = 1e-5
