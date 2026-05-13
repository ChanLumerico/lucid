"""Attention U-Net configuration (Oktay et al., MIDL 2018)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class AttentionUNetConfig(ModelConfig):
    """Configuration for Attention U-Net.

    Extends the standard U-Net architecture (Ronneberger et al., 2015) by
    adding soft attention gates on skip connections.  Each gate computes a
    spatial attention map from the skip feature and the gating signal from
    the decoder, suppressing irrelevant activations before concatenation.

    Architecture overview:
      Encoder: depth × (2×Conv3x3-BN-ReLU + MaxPool2x2)
      Bottleneck: 2×Conv3x3-BN-ReLU
      Decoder: depth × (Upsample/ConvTranspose + AttentionGate + Cat + 2×Conv3x3-BN-ReLU)
      Head: Conv1x1 → num_classes

    Args:
        num_classes:   Number of output segmentation classes.
        in_channels:   Number of input image channels.
        base_channels: Feature channels at the first encoder stage.
                       Doubles at each depth level.
        depth:         Number of encoder/decoder stages (excluding bottleneck).
        bilinear:      If True, use bilinear upsampling; otherwise ConvTranspose2d.
    """

    model_type: ClassVar[str] = "attention_unet"

    num_classes: int = 2
    in_channels: int = 1
    base_channels: int = 64
    depth: int = 4
    bilinear: bool = False
