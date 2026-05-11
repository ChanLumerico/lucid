"""ConvNeXt V2 family — backbone + image classification (Woo et al., 2022).

Paper: "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders"

Key innovation over ConvNeXt V1: Global Response Normalization (GRN) added
inside each MLP block after GELU, enabling masked autoencoder pre-training.
"""

from lucid.models.vision.convnext_v2._config import ConvNeXtV2Config
from lucid.models.vision.convnext_v2._model import (
    ConvNeXtV2,
    ConvNeXtV2ForImageClassification,
)
from lucid.models.vision.convnext_v2._pretrained import (
    convnext_v2_atto,
    convnext_v2_atto_cls,
    convnext_v2_femto,
    convnext_v2_femto_cls,
    convnext_v2_pico,
    convnext_v2_pico_cls,
    convnext_v2_nano,
    convnext_v2_nano_cls,
    convnext_v2_tiny,
    convnext_v2_tiny_cls,
    convnext_v2_small,
    convnext_v2_small_cls,
    convnext_v2_base,
    convnext_v2_base_cls,
    convnext_v2_large,
    convnext_v2_large_cls,
    convnext_v2_huge,
    convnext_v2_huge_cls,
)

__all__ = [
    "ConvNeXtV2Config",
    "ConvNeXtV2",
    "ConvNeXtV2ForImageClassification",
    "convnext_v2_atto",
    "convnext_v2_atto_cls",
    "convnext_v2_femto",
    "convnext_v2_femto_cls",
    "convnext_v2_pico",
    "convnext_v2_pico_cls",
    "convnext_v2_nano",
    "convnext_v2_nano_cls",
    "convnext_v2_tiny",
    "convnext_v2_tiny_cls",
    "convnext_v2_small",
    "convnext_v2_small_cls",
    "convnext_v2_base",
    "convnext_v2_base_cls",
    "convnext_v2_large",
    "convnext_v2_large_cls",
    "convnext_v2_huge",
    "convnext_v2_huge_cls",
]
