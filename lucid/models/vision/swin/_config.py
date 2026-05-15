"""Swin Transformer configuration (Liu et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SwinConfig(ModelConfig):
    """Configuration for Swin Transformer variants (T/S/B/L).

    Canonical variants:
      Swin-T: embed_dim=96,  depths=(2,2,6,2),  num_heads=(3,6,12,24)
      Swin-S: embed_dim=96,  depths=(2,2,18,2), num_heads=(3,6,12,24)
      Swin-B: embed_dim=128, depths=(2,2,18,2), num_heads=(4,8,16,32)
      Swin-L: embed_dim=192, depths=(2,2,18,2), num_heads=(6,12,24,48)

    ``embed_dim`` — channels after patch embedding (stage-1 dim).
      Each subsequent stage doubles: embed_dim → 2× → 4× → 8×.
    ``window_size`` — local attention window (7×7 in the paper).
    ``mlp_ratio`` — MLP hidden expansion (4.0 in the paper).
    ``dropout`` — applied to MLP and attention.
    ``patch_size`` — initial patch merging (4×4 in the paper).
    ``drop_path_rate`` — stochastic-depth max rate; linearly scheduled
        across all blocks of the trunk (Liu et al., 2021 §A).  Paper uses
        0.2 for Swin-T, 0.3 for Swin-S/B/L.
    """

    model_type: ClassVar[str] = "swin"

    image_size: int = 224
    patch_size: int = 4
    in_channels: int = 3
    num_classes: int = 1000
    embed_dim: int = 96
    depths: tuple[int, ...] = (2, 2, 6, 2)
    num_heads: tuple[int, ...] = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    drop_path_rate: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "num_heads", tuple(self.num_heads))
