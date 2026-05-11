"""SKNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SKNetConfig(ModelConfig):
    """Unified config for all SK-ResNet variants (Li et al., 2019).

    Paper: "Selective Kernel Networks"

    The SK block replaces the 3×3 conv in a bottleneck with two parallel
    branches (3×3 and dilated-3×3 with dilation=2, padding=2).  The branches
    are fused via element-wise addition, squeezed with global average pooling,
    and excited through a compact FC → softmax to produce per-branch attention
    weights, which produce a weighted sum of the two branch outputs.

    ``num_paths`` is fixed at 2 in the standard SKNet formulation.
    ``reduction`` controls the bottleneck ratio of the gating FC layer.
    ``cardinality`` / ``width_per_group`` enable ResNeXt-style grouped
    convolutions inside the SK branches (set cardinality=1 for plain SK-ResNet).
    """

    model_type: ClassVar[str] = "sknet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    reduction: int = 16
    num_paths: int = 2
    block_type: str = "bottleneck"
    cardinality: int = 1
    width_per_group: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
