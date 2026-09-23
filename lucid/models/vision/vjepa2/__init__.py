"""V-JEPA 2 — self-supervised video feature prediction."""

from lucid.models.vision.vjepa2._config import VJEPA2Config
from lucid.models.vision.vjepa2._model import (
    VJEPA2ForVideoClassification,
    VJEPA2Model,
    VJEPA2Output,
)
from lucid.models.vision.vjepa2._pretrained import (
    vjepa2_vit_giant,
    vjepa2_vit_giant_384,
    vjepa2_vit_giant_384_cls,
    vjepa2_vit_giant_cls,
    vjepa2_vit_huge,
    vjepa2_vit_huge_cls,
    vjepa2_vit_large,
    vjepa2_vit_large_cls,
)
from lucid.models.vision.vjepa2._weights import (
    VJEPA2ViTGiant384Weights,
    VJEPA2ViTGiantWeights,
    VJEPA2ViTHugeWeights,
    VJEPA2ViTLargeWeights,
)

__all__ = [
    "VJEPA2Config",
    "VJEPA2Model",
    "VJEPA2ForVideoClassification",
    "VJEPA2Output",
    "vjepa2_vit_large",
    "vjepa2_vit_large_cls",
    "vjepa2_vit_huge",
    "vjepa2_vit_huge_cls",
    "vjepa2_vit_giant",
    "vjepa2_vit_giant_cls",
    "vjepa2_vit_giant_384",
    "vjepa2_vit_giant_384_cls",
    "VJEPA2ViTLargeWeights",
    "VJEPA2ViTHugeWeights",
    "VJEPA2ViTGiantWeights",
    "VJEPA2ViTGiant384Weights",
]
