"""CoAtNet family — backbone + image classification (Dai et al., 2021)."""

from lucid.models.vision.coatnet._config import CoAtNetConfig
from lucid.models.vision.coatnet._model import CoAtNet, CoAtNetForImageClassification
from lucid.models.vision.coatnet._pretrained import (
    coatnet_0,
    coatnet_0_cls,
    coatnet_1,
    coatnet_1_cls,
    coatnet_2,
    coatnet_2_cls,
    coatnet_3,
    coatnet_3_cls,
    coatnet_4,
    coatnet_4_cls,
    coatnet_5,
    coatnet_5_cls,
    coatnet_6,
    coatnet_6_cls,
    coatnet_7,
    coatnet_7_cls,
)

__all__ = [
    "CoAtNetConfig",
    "CoAtNet",
    "CoAtNetForImageClassification",
    "coatnet_0",
    "coatnet_0_cls",
    "coatnet_1",
    "coatnet_1_cls",
    "coatnet_2",
    "coatnet_2_cls",
    "coatnet_3",
    "coatnet_3_cls",
    "coatnet_4",
    "coatnet_4_cls",
    "coatnet_5",
    "coatnet_5_cls",
    "coatnet_6",
    "coatnet_6_cls",
    "coatnet_7",
    "coatnet_7_cls",
]
