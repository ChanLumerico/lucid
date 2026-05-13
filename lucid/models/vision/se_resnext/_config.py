"""SE-ResNeXt configuration."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class SEResNeXtConfig(ModelConfig):
    """Configuration for SE-ResNeXt variants.

    Combines ResNeXt's grouped convolution with Squeeze-and-Excitation gates
    inside each bottleneck block.

    ``cardinality``  — number of groups in the 3×3 grouped conv.
    ``base_width``   — base width per group (width_per_group in ResNeXt terms).
    ``se_reduction`` — SE reduction ratio (applied to out_ch * expansion).
    """

    model_type: ClassVar[str] = "se_resnext"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    cardinality: int = 32
    base_width: int = 4  # base_width per group
    se_reduction: int = 16
    dropout: float = 0.0
