"""ResNeSt configuration (Zhang et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ResNeStConfig(ModelConfig):
    """Configuration for ResNeSt.

    ``layers``  — per-stage block repetition counts.
    ``radix``   — number of split branches in SplitAttention conv.
    """

    model_type: ClassVar[str] = "resnest"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    radix: int = 2
    dropout: float = 0.0
    zero_init_residual: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
