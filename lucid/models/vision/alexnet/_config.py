"""AlexNet configuration (Krizhevsky, Sutskever & Hinton, 2012)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class AlexNetConfig(ModelConfig):
    """Configuration for AlexNet.

    The canonical AlexNet takes 3×224×224 inputs and produces 4096-dim
    embeddings before the final classifier.

    ``dropout`` controls the two dropout layers inside the classifier
    (0.5 in the original paper).
    """

    model_type: ClassVar[str] = "alexnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.5
