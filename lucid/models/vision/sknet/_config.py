"""SKNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SKNetConfig(ModelConfig):
    """Unified config for all SK-ResNet variants (Li et al., 2019).

    Paper: "Selective Kernel Networks"

    Architecture (SK-ResNet-50 defaults):
      expansion = 2  (bottleneck: in → mid → mid*2)
      Stage outputs: 128, 256, 512, 1024  (not ResNet's 256/512/1024/2048)
      Final classifier: FC(1024, num_classes)

      ``cardinality`` = G in the paper (number of groups in SK branches).
      SK-ResNet-50 uses G=32 (default). The 1×1 projection convs are always
      ungrouped — only the SK branch convolutions use cardinality groups.
      ``reduction`` controls the squeeze ratio of the gating FC (r=16 default,
      minimum channel dim clamped to 32).
    """

    model_type: ClassVar[str] = "sknet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    reduction: int = 16
    cardinality: int = 32

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
