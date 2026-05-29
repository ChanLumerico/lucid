"""InceptionNeXt — Yu et al., 2023."""

from lucid.models.vision.inception_next._config import InceptionNeXtConfig
from lucid.models.vision.inception_next._model import (
    InceptionNeXt,
    InceptionNeXtForImageClassification,
)
from lucid.models.vision.inception_next._pretrained import (
    inception_next_base,
    inception_next_base_cls,
    inception_next_small,
    inception_next_small_cls,
    inception_next_tiny,
    inception_next_tiny_cls,
)
from lucid.models.vision.inception_next._weights import (
    InceptionNeXtBaseWeights,
    InceptionNeXtSmallWeights,
    InceptionNeXtTinyWeights,
)

__all__ = [
    "InceptionNeXtConfig",
    "InceptionNeXt",
    "InceptionNeXtForImageClassification",
    "inception_next_tiny",
    "inception_next_tiny_cls",
    "inception_next_small",
    "inception_next_small_cls",
    "inception_next_base",
    "inception_next_base_cls",
    "InceptionNeXtTinyWeights",
    "InceptionNeXtSmallWeights",
    "InceptionNeXtBaseWeights",
]
