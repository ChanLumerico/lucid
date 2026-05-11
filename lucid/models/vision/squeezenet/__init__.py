"""SqueezeNet family — backbone + image classification."""

from lucid.models.vision.squeezenet._config import SqueezeNetConfig
from lucid.models.vision.squeezenet._model import (
    SqueezeNet,
    SqueezeNetForImageClassification,
)
from lucid.models.vision.squeezenet._pretrained import (
    squeezenet_1_0,
    squeezenet_1_0_cls,
    squeezenet_1_1,
    squeezenet_1_1_cls,
)

__all__ = [
    "SqueezeNetConfig",
    "SqueezeNet",
    "SqueezeNetForImageClassification",
    "squeezenet_1_0",
    "squeezenet_1_0_cls",
    "squeezenet_1_1",
    "squeezenet_1_1_cls",
]
