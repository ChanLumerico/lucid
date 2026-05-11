"""ConvNeXt configuration (Liu et al., 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ConvNeXtConfig(ModelConfig):
    """Configuration for ConvNeXt T/S/B/L/XL.

    Canonical variants:
      ConvNeXt-T : depths=(3,3,9,3),   dims=(96,192,384,768)
      ConvNeXt-S : depths=(3,3,27,3),  dims=(96,192,384,768)
      ConvNeXt-B : depths=(3,3,27,3),  dims=(128,256,512,1024)
      ConvNeXt-L : depths=(3,3,27,3),  dims=(192,384,768,1536)
      ConvNeXt-XL: depths=(3,3,27,3),  dims=(256,512,1024,2048)

    Each ConvNeXt block:
      DWConv(7×7) → LN → Linear(dim → 4×dim) → GELU → Linear(4×dim → dim)
      with a layer-scale parameter (γ) initialised to 1e-6.

    ``layer_scale_init`` — initial value for the per-channel scale (1e-6).
    ``dropout`` — stochastic depth rate (uniform across all blocks in paper).
    """

    model_type: ClassVar[str] = "convnext"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (3, 3, 9, 3)
    dims: tuple[int, ...] = (96, 192, 384, 768)
    layer_scale_init: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "dims", tuple(self.dims))
