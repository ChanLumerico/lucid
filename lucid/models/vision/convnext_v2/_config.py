"""ConvNeXt V2 configuration (Woo et al., 2022).

Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"
"""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ConvNeXtV2Config(ModelConfig):
    """Configuration for ConvNeXt V2 variants.

    ConvNeXt V2 is ConvNeXt V1 with Global Response Normalization (GRN) added
    inside the MLP block, after the GELU activation.

    Canonical variants (from Table 1 of the paper):
      atto  : depths=(2,2,6,2),  dims=(40,80,160,320)
      femto : depths=(2,2,6,2),  dims=(48,96,192,384)
      pico  : depths=(2,2,6,2),  dims=(64,128,256,512)
      nano  : depths=(2,2,8,2),  dims=(80,160,320,640)
      tiny  : depths=(3,3,9,3),  dims=(96,192,384,768)
      small : depths=(3,3,27,3), dims=(96,192,384,768)
      base  : depths=(3,3,27,3), dims=(128,256,512,1024)
      large : depths=(3,3,27,3), dims=(192,384,768,1536)
      huge  : depths=(3,3,27,3), dims=(352,704,1408,2816)

    ``layer_scale_init`` — initial value for per-channel layer scale (1e-6).
    ``dropout`` — stochastic depth rate (uniform across all blocks).
    """

    model_type: ClassVar[str] = "convnext_v2"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (3, 3, 9, 3)
    dims: tuple[int, ...] = (96, 192, 384, 768)
    layer_scale_init: float = 1e-6
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "dims", tuple(self.dims))
