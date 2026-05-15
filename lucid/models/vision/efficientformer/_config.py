"""EfficientFormer configuration (Li et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class EfficientFormerConfig(ModelConfig):
    """Configuration for EfficientFormer variants.

    Canonical variants:
      EfficientFormer-L1: depths=(3,2,6,4), embed_dims=(48,96,224,448)
      EfficientFormer-L3: depths=(4,4,12,6), embed_dims=(64,128,320,512)
      EfficientFormer-L7: depths=(6,6,18,8), embed_dims=(96,192,384,768)

    Stages 1–3 use MetaFormer-style pooling attention (no QKV).
    Stage 4 uses standard multi-head attention for global reasoning.

    ``embed_dims`` — output channels per stage.
    ``depths``     — blocks per stage.
    ``mlp_ratios`` — MLP expansion ratio per stage.
    """

    model_type: ClassVar[str] = "efficientformer"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (3, 2, 6, 4)
    embed_dims: tuple[int, ...] = (48, 96, 224, 448)
    mlp_ratios: tuple[float, ...] = (4.0, 4.0, 4.0, 4.0)
    # Regularization knobs (paper §4.1):
    #   drop_path_rate    — max stochastic depth rate (linear schedule across trunk).
    #                       Paper uses 0.0 for L1, 0.1 for L3, 0.2 for L7.
    #   layer_scale_init  — γ initialization for LayerScale on every residual
    #                       branch (paper appendix: 1e-5).
    drop_path_rate: float = 0.0
    layer_scale_init: float = 1e-5

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "embed_dims", tuple(self.embed_dims))
        object.__setattr__(self, "mlp_ratios", tuple(self.mlp_ratios))
