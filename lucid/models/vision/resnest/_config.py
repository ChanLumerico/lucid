"""ResNeSt configuration (Zhang et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ResNeStConfig(ModelConfig):
    """Configuration for ResNeSt.

    ``layers``           — per-stage block repetition counts.
    ``radix``            — number of split branches in SplitAttn conv.
    ``groups``           — cardinality (number of convolution groups).
    ``avg_down``         — use AvgPool + 1×1 Conv for downsampling shortcuts.
    ``avd``              — use averaged downsampling (AvgPool) around SplitAttn.
    ``avd_first``        — place the AvgPool before (True) or after (False) SplitAttn.
    ``stem_width``       — channel width of each deep-stem conv (output = stem_width*2).
    ``deep_stem``        — use a 3-convolution deep stem instead of a single 7×7 conv.
    """

    model_type: ClassVar[str] = "resnest"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    radix: int = 2
    groups: int = 1
    avg_down: bool = True
    avd: bool = True
    avd_first: bool = False
    stem_width: int = 32
    deep_stem: bool = True
    dropout: float = 0.0
    zero_init_residual: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
