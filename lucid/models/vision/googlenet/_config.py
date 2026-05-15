"""GoogLeNet (Inception v1) configuration (Szegedy et al., 2014)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class GoogLeNetConfig(ModelConfig):
    """Configuration for GoogLeNet (Inception v1).

    ``aux_logits`` enables the two auxiliary classifiers that were used
    during training in the original paper.  They are ignored at inference
    (eval mode); the auxiliary loss is typically weighted 0.3 in the total
    training objective.

    ``dropout`` — main classifier dropout rate (0.4 in the paper).
    ``aux_dropout`` — auxiliary classifier dropout rate (0.7 in the paper).
    """

    model_type: ClassVar[str] = "googlenet"

    num_classes: int = 1000
    in_channels: int = 3
    aux_logits: bool = True
    dropout: float = 0.4
    aux_dropout: float = 0.7
