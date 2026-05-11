"""Inception-ResNet v2 configuration (Szegedy et al., 2016)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class InceptionResNetConfig(ModelConfig):
    """Configuration for Inception-ResNet v2.

    Uses the same stem as Inception-v4, followed by:
      - Mixed_5b (192 → 320)
      - 10× Block35  (scale_a, default 0.17)
      - Mixed_6a / Reduction-A (320 → 1088)
      - 20× Block17  (scale_b, default 0.10)
      - Mixed_7a / Reduction-B (1088 → 2080)
      -  9× Block8   (scale_c, default 0.20) + 1× Block8 (no ReLU)
      - Conv2d_7b 1×1 projection (2080 → 1536)
      - AdaptiveAvgPool → Dropout → FC(1536, num_classes)

    ``scale_a`` — residual scale for Block35 (paper suggests 0.17).
    ``scale_b`` — residual scale for Block17 (paper suggests 0.10).
    ``scale_c`` — residual scale for Block8  (paper suggests 0.20).
    ``dropout`` — head dropout rate (0.2 in the paper).
    """

    model_type: ClassVar[str] = "inception_resnet"

    num_classes: int = 1000
    in_channels: int = 3
    dropout: float = 0.2
    scale_a: float = 0.17  # Block35 (Inception-A)
    scale_b: float = 0.10  # Block17 (Inception-B)
    scale_c: float = 0.20  # Block8  (Inception-C)
