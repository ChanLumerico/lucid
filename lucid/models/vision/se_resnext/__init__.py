"""SE-ResNeXt family — backbone + image classification."""

from lucid.models.vision.se_resnext._config import SEResNeXtConfig
from lucid.models.vision.se_resnext._model import (
    SEResNeXt,
    SEResNeXtForImageClassification,
)
from lucid.models.vision.se_resnext._pretrained import (
    se_resnext_50_32x4d,
    se_resnext_50_32x4d_cls,
    se_resnext_101_32x4d,
    se_resnext_101_32x4d_cls,
)

__all__ = [
    "SEResNeXtConfig",
    "SEResNeXt",
    "SEResNeXtForImageClassification",
    "se_resnext_50_32x4d",
    "se_resnext_50_32x4d_cls",
    "se_resnext_101_32x4d",
    "se_resnext_101_32x4d_cls",
]
