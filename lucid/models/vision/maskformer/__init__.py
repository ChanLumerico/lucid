"""MaskFormer (Cheng et al., NeurIPS 2021).

Paper: "Per-Pixel Classification is Not All You Need for Semantic Segmentation"
"""

from lucid.models.vision.maskformer._config import MaskFormerConfig
from lucid.models.vision.maskformer._model import MaskFormerForSemanticSegmentation
from lucid.models.vision.maskformer._pretrained import (
    maskformer_resnet50,
    maskformer_resnet101,
)

__all__ = [
    "MaskFormerConfig",
    "MaskFormerForSemanticSegmentation",
    "maskformer_resnet50",
    "maskformer_resnet101",
]
