"""NFNet family — Brock et al., 2021."""

from lucid.models.vision.nfnet._config import NFNetConfig
from lucid.models.vision.nfnet._model import NFNet, NFNetForImageClassification
from lucid.models.vision.nfnet._pretrained import (
    nfnet_f0,
    nfnet_f0_cls,
    nfnet_f1,
    nfnet_f1_cls,
    nfnet_f2,
    nfnet_f2_cls,
    nfnet_f3,
    nfnet_f3_cls,
)

__all__ = [
    "NFNetConfig",
    "NFNet",
    "NFNetForImageClassification",
    "nfnet_f0",
    "nfnet_f0_cls",
    "nfnet_f1",
    "nfnet_f1_cls",
    "nfnet_f2",
    "nfnet_f2_cls",
    "nfnet_f3",
    "nfnet_f3_cls",
]
