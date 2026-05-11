"""InceptionNeXt configuration (Yu et al., 2023)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class InceptionNeXtConfig(ModelConfig):
    """Configuration for InceptionNeXt variants.

    Canonical variants:
      InceptionNeXt-T: depths=(3,3,9,3),   dims=(96,192,384,768)
      InceptionNeXt-S: depths=(3,3,27,3),  dims=(96,192,384,768)
      InceptionNeXt-B: depths=(3,3,27,3),  dims=(128,256,512,1024)

    Key: DWConv in ConvNeXt block is replaced by ``InceptionDWConv2d``,
    which decomposes large kernels into parallel branches:
      - square small 3×3 DWConv
      - horizontal 1×K + vertical K×1 DWConv (band convolution)
      - additional small 3×3 for high-frequency

    ``depths``     — blocks per stage (matches ConvNeXt).
    ``dims``       — channel dimensions per stage.
    ``band_kernel`` — K for band (1×K / K×1) convolutions.
    """

    model_type: ClassVar[str] = "inception_next"

    num_classes: int = 1000
    in_channels: int = 3
    depths: tuple[int, ...] = (3, 3, 9, 3)
    dims: tuple[int, ...] = (96, 192, 384, 768)
    band_kernel: int = 11

    def __post_init__(self) -> None:
        object.__setattr__(self, "depths", tuple(self.depths))
        object.__setattr__(self, "dims", tuple(self.dims))
