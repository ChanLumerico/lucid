"""Mask2Former (Cheng et al., CVPR 2022).

Paper: "Masked-attention Mask Transformer for Universal Image Segmentation"
"""

from lucid.models.vision.mask2former._config import Mask2FormerConfig
from lucid.models.vision.mask2former._model import Mask2FormerForSemanticSegmentation
from lucid.models.vision.mask2former._pretrained import (
    mask2former_resnet50,
    mask2former_resnet101,
    mask2former_swin_tiny,
    mask2former_swin_small,
    mask2former_swin_base,
    mask2former_swin_large,
)

__all__ = [
    "Mask2FormerConfig",
    "Mask2FormerForSemanticSegmentation",
    "mask2former_resnet50",
    "mask2former_resnet101",
    "mask2former_swin_tiny",
    "mask2former_swin_small",
    "mask2former_swin_base",
    "mask2former_swin_large",
]
