"""DenseNet family — Huang et al., CVPR 2017."""

from lucid.models.vision.densenet._config import DenseNetConfig
from lucid.models.vision.densenet._model import DenseNet, DenseNetForImageClassification
from lucid.models.vision.densenet._pretrained import (
    densenet_121,
    densenet_121_cls,
    densenet_161,
    densenet_161_cls,
    densenet_169,
    densenet_169_cls,
    densenet_201,
    densenet_201_cls,
    densenet_264,
    densenet_264_cls,
)
from lucid.models.vision.densenet._weights import (
    DenseNet121Weights,
    DenseNet161Weights,
    DenseNet169Weights,
    DenseNet201Weights,
)

__all__ = [
    "DenseNetConfig",
    "DenseNet",
    "DenseNetForImageClassification",
    "DenseNet121Weights",
    "DenseNet161Weights",
    "DenseNet169Weights",
    "DenseNet201Weights",
    "densenet_121",
    "densenet_121_cls",
    "densenet_161",
    "densenet_161_cls",
    "densenet_169",
    "densenet_169_cls",
    "densenet_201",
    "densenet_201_cls",
    "densenet_264",
    "densenet_264_cls",
]
