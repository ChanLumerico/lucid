"""PVT — Pyramid Vision Transformer v2 (Wang et al., 2021/2022)."""

from lucid.models.vision.pvt._config import PVTConfig
from lucid.models.vision.pvt._model import PVT, PVTForImageClassification
from lucid.models.vision.pvt._pretrained import (
    pvt_tiny,
    pvt_tiny_cls,
    pvt_v2_b0,
    pvt_v2_b0_cls,
    pvt_v2_b1,
    pvt_v2_b1_cls,
    pvt_v2_b2,
    pvt_v2_b2_cls,
    pvt_v2_b3,
    pvt_v2_b3_cls,
    pvt_v2_b4,
    pvt_v2_b4_cls,
    pvt_v2_b5,
    pvt_v2_b5_cls,
)

__all__ = [
    "PVTConfig",
    "PVT",
    "PVTForImageClassification",
    # Canonical variants (B0–B5)
    "pvt_v2_b0",
    "pvt_v2_b0_cls",
    "pvt_v2_b1",
    "pvt_v2_b1_cls",
    "pvt_v2_b2",
    "pvt_v2_b2_cls",
    "pvt_v2_b3",
    "pvt_v2_b3_cls",
    "pvt_v2_b4",
    "pvt_v2_b4_cls",
    "pvt_v2_b5",
    "pvt_v2_b5_cls",
    # Backwards-compat aliases
    "pvt_tiny",
    "pvt_tiny_cls",
]
