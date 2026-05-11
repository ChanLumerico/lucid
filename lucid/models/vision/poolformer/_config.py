"""PoolFormer configuration (Yu et al., 2022).

Paper: "MetaFormer is Actually What You Need for Vision"
"""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class PoolFormerConfig(ModelConfig):
    """Configuration for PoolFormer variants.

    Canonical variants (paper Table 1):
      PoolFormer-S12: layers=(2,2,6,2),  embed_dims=(64,128,320,512)
      PoolFormer-S24: layers=(4,4,12,4), embed_dims=(64,128,320,512)
      PoolFormer-S36: layers=(6,6,18,6), embed_dims=(64,128,320,512)
      PoolFormer-M36: layers=(6,6,18,6), embed_dims=(96,192,384,768)
      PoolFormer-M48: layers=(8,8,24,8), embed_dims=(96,192,384,768)

    ``layers`` — number of PoolFormer blocks in each of the 4 stages.
    ``embed_dims`` — channel count for each of the 4 stages.
    ``mlp_ratio`` — MLP hidden-dim expansion factor (default 4.0).
    ``pool_size`` — kernel size for the average-pooling token mixer (default 3).
    ``dropout`` — dropout rate applied inside the MLP (default 0.0).
    ``layer_scale_init`` — initial value for LayerScale parameters (default 1e-5).
    """

    model_type: ClassVar[str] = "poolformer"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (2, 2, 6, 2)
    embed_dims: tuple[int, ...] = (64, 128, 320, 512)
    mlp_ratio: float = 4.0
    pool_size: int = 3
    dropout: float = 0.0
    layer_scale_init: float = 1e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "embed_dims", tuple(self.embed_dims))
