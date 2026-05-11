"""DeiT family — backbone + image classification (Touvron et al., 2021).

Paper: "Training data-efficient image transformers & distillation through
attention"

DeiT extends ViT with a distillation token that enables knowledge distillation
from a convolutional teacher at training time.  At inference, predictions from
the class token and the distillation token are averaged.
"""

from lucid.models.vision.deit._config import DeiTConfig
from lucid.models.vision.deit._model import DeiT, DeiTForImageClassification
from lucid.models.vision.deit._pretrained import (
    deit_tiny,
    deit_tiny_cls,
    deit_small,
    deit_small_cls,
    deit_base,
    deit_base_cls,
    deit_base_patch32,
    deit_base_patch32_cls,
)

__all__ = [
    "DeiTConfig",
    "DeiT",
    "DeiTForImageClassification",
    "deit_tiny",
    "deit_tiny_cls",
    "deit_small",
    "deit_small_cls",
    "deit_base",
    "deit_base_cls",
    "deit_base_patch32",
    "deit_base_patch32_cls",
]
