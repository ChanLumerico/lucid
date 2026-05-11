"""CoAtNet family — backbone + image classification (Dai et al., 2021)."""

from lucid.models.vision.coatnet._config import CoAtNetConfig
from lucid.models.vision.coatnet._model import CoAtNet, CoAtNetForImageClassification
from lucid.models.vision.coatnet._pretrained import coatnet_0, coatnet_0_cls

__all__ = [
    "CoAtNetConfig",
    "CoAtNet",
    "CoAtNetForImageClassification",
    "coatnet_0",
    "coatnet_0_cls",
]
