"""Inception v3 configuration (Szegedy et al., 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class InceptionConfig(ModelConfig):
    """Configuration for Inception v3.

    ``aux_logits`` enables the auxiliary classifier that attaches after the
    second InceptionC block (used during training with weight 0.4).
    ``dropout`` — main head dropout rate (0.5 in the paper).
    ``version`` — only ``"v3"`` is supported in this module.
    """

    model_type: ClassVar[str] = "inception_v3"

    num_classes: int = 1000
    in_channels: int = 3
    version: str = "v3"
    aux_logits: bool = True
    dropout: float = 0.5
