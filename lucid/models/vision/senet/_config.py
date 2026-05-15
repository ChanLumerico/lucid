"""SENet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SENetConfig(ModelConfig):
    """Unified config for all SE-ResNet variants (Hu et al., 2017).

    Paper: "Squeeze-and-Excitation Networks"

    ``block_type`` selects BasicBlock (SE-ResNet-18/34) or Bottleneck
    (SE-ResNet-50/101/152).  ``layers`` is the per-stage repetition count.
    ``reduction`` is the channel reduction ratio in the SE block (default 16).
    """

    model_type: ClassVar[str] = "senet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    reduction: int = 16
    block_type: str = "bottleneck"

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
