"""CSPNet configuration dataclass (Wang et al., 2019)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class CSPNetConfig(ModelConfig):
    """Unified config for CSPResNet variants.

    ``layers`` — per-stage block counts (4 stages total).
    ``channels`` — base channel widths per stage before CSP branching.
    """

    model_type: ClassVar[str] = "cspnet"

    num_classes: int = 1000
    in_channels: int = 3
    layers: tuple[int, ...] = (3, 3, 5, 2)
    channels: tuple[int, ...] = (64, 128, 256, 512)

    def __post_init__(self) -> None:
        object.__setattr__(self, "layers", tuple(self.layers))
        object.__setattr__(self, "channels", tuple(self.channels))
