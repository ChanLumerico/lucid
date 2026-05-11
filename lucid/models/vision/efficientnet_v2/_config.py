"""EfficientNetV2 configuration (Tan & Le, 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class EfficientNetV2Config(ModelConfig):
    """Configuration for EfficientNetV2 S/M/L/XL variants.

    Each variant is defined by a fixed stage table (block_type, expand_ratio,
    kernel, stride, in_ch, out_ch, num_blocks, se_ratio).  Unlike V1, V2 uses
    FusedMBConv for early stages and standard MBConv (with SE) for later stages.

    ``drop_connect_rate`` — stochastic depth maximum rate; applied linearly
    across blocks.  Set to 0.0 to disable.
    """

    model_type: ClassVar[str] = "efficientnet_v2"

    num_classes: int = 1000
    in_channels: int = 3
    variant: str = "small"  # "small", "medium", "large", "xlarge"
    dropout: float = 0.2
    drop_connect_rate: float = 0.2
