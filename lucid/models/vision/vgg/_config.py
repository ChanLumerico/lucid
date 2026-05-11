"""VGG configuration (Simonyan & Zisserman, 2014)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class VGGConfig(ModelConfig):
    """Configuration for all VGG variants (A/B/D/E ≡ 11/13/16/19).

    ``arch`` encodes the per-block conv counts:
      - ``(1, 1, 2, 2, 2)`` → VGG-11
      - ``(2, 2, 2, 2, 2)`` → VGG-13
      - ``(2, 2, 3, 3, 3)`` → VGG-16
      - ``(2, 2, 4, 4, 4)`` → VGG-19

    ``batch_norm`` enables BatchNorm after each Conv+ReLU pair (VGG-BN).
    ``dropout`` applies to the two 4096-dim FC layers (0.5 in the paper).
    """

    model_type: ClassVar[str] = "vgg"

    num_classes: int = 1000
    in_channels: int = 3
    arch: tuple[int, ...] = (2, 2, 3, 3, 3)  # VGG-16 default
    batch_norm: bool = False
    dropout: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "arch", tuple(self.arch))
