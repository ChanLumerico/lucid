"""MaxViT — Multi-Axis Vision Transformer (Tu et al., 2022)."""

from lucid.models.vision.maxvit._config import MaxViTConfig
from lucid.models.vision.maxvit._model import MaxViT, MaxViTForImageClassification
from lucid.models.vision.maxvit._pretrained import maxvit_t, maxvit_t_cls

__all__ = [
    "MaxViTConfig",
    "MaxViT",
    "MaxViTForImageClassification",
    "maxvit_t",
    "maxvit_t_cls",
]
