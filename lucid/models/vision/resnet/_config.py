"""ResNet configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ResNetConfig(ModelConfig):
    """Unified config for all ResNet variants.

    ``block_type`` selects BasicBlock (18/34) or Bottleneck (50/101/152).
    ``layers`` is the per-stage repetition count, e.g. ``(3, 4, 6, 3)`` for ResNet-50.
    """

    model_type: ClassVar[str] = "resnet"

    num_classes: int = 1000
    in_channels: int = 3
    block_type: str = "bottleneck"
    layers: tuple[int, ...] = (3, 4, 6, 3)
    stem_channels: int = 64
    hidden_sizes: tuple[int, ...] = (64, 128, 256, 512)
    dropout: float = 0.0
    zero_init_residual: bool = False

    def __post_init__(self) -> None:
        # JSON round-trips lists; coerce back to tuples for frozen-dataclass fields.
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "hidden_sizes", tuple(self.hidden_sizes))
