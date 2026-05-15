"""ResNeXt configuration dataclass."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class ResNeXtConfig(ModelConfig):
    """Unified config for all ResNeXt variants (Xie et al., 2017).

    ResNeXt extends ResNet by replacing the plain 3×3 conv in each bottleneck
    with a grouped convolution of ``cardinality`` groups, where each group
    handles ``width_per_group`` channels.

    ``layers`` is the per-stage repetition count, e.g. ``(3, 4, 6, 3)`` for
    ResNeXt-50.  ``cardinality`` and ``width_per_group`` jointly determine the
    intermediate width inside each bottleneck.
    """

    model_type: ClassVar[str] = "resnext"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 4, 6, 3)
    cardinality: int = 32
    width_per_group: int = 4
    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
