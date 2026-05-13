"""FCN configuration (Long et al., CVPR 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class FCNConfig(ModelConfig):
    """Configuration for Fully Convolutional Network (FCN).

    FCN adapts a classification CNN backbone (ResNet) into a dense
    predictor by replacing fully-connected layers with convolutions and
    adding upsampling to restore spatial resolution.

    Architecture overview:
      ResNet backbone with dilated convolutions (layer3 dilation=2,
      layer4 dilation=4) → FCN head (3×3 conv + BN + ReLU + 1×1 conv)
      → bilinear upsample to input resolution.

      An auxiliary head on layer3 output provides additional supervision
      during training.

    Args:
        num_classes:         Number of output segmentation classes.
        in_channels:         Number of input image channels.
        backbone:            Backbone descriptor label ("resnet50" or "resnet101").
        variant:             FCN variant string ("fcn32s", "fcn16s", "fcn8s").
        classifier_hidden_channels:
                             Hidden channels in the FCN segmentation head.
        aux_hidden_channels: Hidden channels in the auxiliary head.
        dropout:             Dropout probability in the heads.
    """

    model_type: ClassVar[str] = "fcn"

    num_classes: int = 21
    in_channels: int = 3
    backbone: str = "resnet50"
    variant: str = "fcn32s"
    classifier_hidden_channels: int = 512
    aux_hidden_channels: int = 256
    dropout: float = 0.1
