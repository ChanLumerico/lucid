"""MaxViT — Multi-Axis Vision Transformer (Tu et al., 2022)."""

from lucid.models.vision.maxvit._config import MaxViTConfig
from lucid.models.vision.maxvit._model import MaxViT, MaxViTForImageClassification
from lucid.models.vision.maxvit._pretrained import (
    maxvit_tiny,
    maxvit_tiny_cls,
    maxvit_small,
    maxvit_small_cls,
    maxvit_base,
    maxvit_base_cls,
    maxvit_large,
    maxvit_large_cls,
    maxvit_xlarge,
    maxvit_xlarge_cls,
)

__all__ = [
    "MaxViTConfig",
    "MaxViT",
    "MaxViTForImageClassification",
    "maxvit_tiny",
    "maxvit_tiny_cls",
    "maxvit_small",
    "maxvit_small_cls",
    "maxvit_base",
    "maxvit_base_cls",
    "maxvit_large",
    "maxvit_large_cls",
    "maxvit_xlarge",
    "maxvit_xlarge_cls",
]
