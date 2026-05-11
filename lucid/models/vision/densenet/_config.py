"""DenseNet configuration (Huang et al., 2016)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class DenseNetConfig(ModelConfig):
    """Configuration for all DenseNet variants (121/169/201/264).

    ``growth_rate`` (k) — number of feature maps each dense layer contributes.
    ``block_config`` — number of dense layers per block (4 blocks total).
    ``num_init_features`` — channels after the initial conv stem.
    ``bn_size`` — bottleneck expansion factor (each layer uses bn_size*k filters
      in its 1×1 branch before the 3×3 branch).
    ``dropout_rate`` — dropout after each dense layer (0 = disabled).
    ``memory_efficient`` — toggles checkpointing in dense blocks (unused here,
      kept for API parity with other frameworks).
    """

    model_type: ClassVar[str] = "densenet"

    num_classes: int = 1000
    in_channels: int = 3
    growth_rate: int = 32
    block_config: tuple[int, ...] = (6, 12, 24, 16)  # DenseNet-121
    num_init_features: int = 64
    bn_size: int = 4
    dropout_rate: float = 0.0
    memory_efficient: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "block_config", tuple(self.block_config))
