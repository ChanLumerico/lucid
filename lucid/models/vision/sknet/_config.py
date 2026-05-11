"""SKNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SKNetConfig(ModelConfig):
    """Unified config for all SK-ResNet variants (Li et al., 2019).

    Paper: "Selective Kernel Networks"

    Architecture is identical to ResNet-50 (expansion=4, stages output
    256/512/1024/2048 channels) except each 3×3 conv in the bottleneck
    is replaced by a SelectiveKernel unit with two parallel branches
    (3×3 + 3×3 dilated-2, mimicking 5×5 receptive field).

    Key hyper-parameters:

    ``layers``
        Number of bottleneck blocks per stage (default ResNet-50 = 3/4/6/3).

    ``cardinality``
        Number of groups for the SK branch convolutions (G in the paper).
        Also used in the ResNeXt-style width formula:
          width = int(planes * (base_width / 64)) * cardinality
        Set to 1 for plain SK-ResNet (default).

    ``base_width``
        Base channel multiplier for the ResNeXt width formula.
        64 → plain ResNet widths (64/128/256/512 at each stage).
        4 with cardinality=32 → SK-ResNeXt-50 32×4d (SKNet-50 from paper).

    ``split_input``
        If True (timm default), each SK branch receives half the input
        channels, keeping the param count similar to a single grouped conv.

    ``rd_ratio``
        Reduction ratio for the SelectiveKernelAttn bottleneck.

    ``rd_divisor``
        Divisor for rounding the attention channel count.
    """

    model_type: ClassVar[str] = "sknet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    cardinality: int = 1
    base_width: int = 64
    split_input: bool = True
    rd_ratio: float = 1.0 / 16
    rd_divisor: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
