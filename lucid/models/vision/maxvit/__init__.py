"""MaxViT — Multi-Axis Vision Transformer (Tu et al., 2022)."""

from lucid.models.vision.maxvit._config import MaxViTConfig
from lucid.models.vision.maxvit._model import MaxViT, MaxViTForImageClassification
from lucid.models.vision.maxvit._pretrained import (
    maxvit_t,
    maxvit_t_cls,
    maxvit_s,
    maxvit_s_cls,
    maxvit_b,
    maxvit_b_cls,
    maxvit_l,
    maxvit_l_cls,
    maxvit_xl,
    maxvit_xl_cls,
)

__all__ = [
    "MaxViTConfig",
    "MaxViT",
    "MaxViTForImageClassification",
    "maxvit_t",
    "maxvit_t_cls",
    "maxvit_s",
    "maxvit_s_cls",
    "maxvit_b",
    "maxvit_b_cls",
    "maxvit_l",
    "maxvit_l_cls",
    "maxvit_xl",
    "maxvit_xl_cls",
]
