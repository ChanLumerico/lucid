"""U-Net configuration (Ronneberger et al., MICCAI 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class UNetConfig(ModelConfig):
    """Configuration for U-Net.

    U-Net is a fully convolutional encoder-decoder architecture with skip
    connections between corresponding encoder and decoder stages.  The
    architecture was originally proposed for biomedical image segmentation.

    Architecture overview:
      Encoder: depth × (DoubleConv + MaxPool2d) — halves spatial resolution.
      Bottleneck: DoubleConv at the deepest level.
      Decoder: depth × (Upsample + skip-cat + DoubleConv).
      Head: Conv2d(base_channels, num_classes, 1).

    Channel schedule (base_channels=64, depth=4):
      Encoder: 64 → 128 → 256 → 512
      Bottleneck: 1024
      Decoder: 512 → 256 → 128 → 64

    Args:
        num_classes:   Number of output segmentation classes.
        in_channels:   Number of input image channels.
        base_channels: Feature channels at the first encoder stage.
                       Doubles at each depth level.
        depth:         Number of encoder/decoder stages (excluding bottleneck).
        bilinear:      If True, use bilinear upsampling + Conv2d;
                       otherwise use ConvTranspose2d for learned upsampling.
        dropout:       Dropout probability applied in DoubleConv blocks.
    """

    model_type: ClassVar[str] = "unet"

    num_classes: int = 2
    in_channels: int = 1
    base_channels: int = 64
    depth: int = 4
    bilinear: bool = False
    dropout: float = 0.0
