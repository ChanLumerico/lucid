"""ResNet model family — feature-extracting backbones and image classifiers.

Implements the residual-learning architecture from He et al., "Deep Residual
Learning for Image Recognition", CVPR 2016 (arXiv:1512.03385), plus a few
widely-cited follow-up variants:

- **Canonical ResNets** — :func:`resnet_18`, :func:`resnet_34`,
  :func:`resnet_50`, :func:`resnet_101`, :func:`resnet_152` and their
  ``*_cls`` classification counterparts.
- **Wide ResNets** (Zagoruyko & Komodakis, BMVC 2016) —
  :func:`wide_resnet_50`, :func:`wide_resnet_101`.
- **Deep bottleneck variants** — :func:`resnet_200`, :func:`resnet_269`.

Every variant shares the same architecture-specifying dataclass
:class:`ResNetConfig`.  Use the factory functions for paper-cited
configurations; pass ``**overrides`` to tweak individual fields without
writing a config by hand.
"""

from lucid.models.vision.resnet._config import ResNetConfig
from lucid.models.vision.resnet._model import ResNet, ResNetForImageClassification
from lucid.weights.vision.resnet import ResNet18Weights
from lucid.models.vision.resnet._pretrained import (
    resnet_18,
    resnet_18_cls,
    resnet_34,
    resnet_34_cls,
    resnet_50,
    resnet_50_cls,
    resnet_101,
    resnet_101_cls,
    resnet_152,
    resnet_152_cls,
    wide_resnet_50,
    wide_resnet_50_cls,
    wide_resnet_101,
    wide_resnet_101_cls,
    resnet_200,
    resnet_200_cls,
    resnet_269,
    resnet_269_cls,
)

__all__ = [
    "ResNetConfig",
    "ResNet",
    "ResNetForImageClassification",
    "ResNet18Weights",
    "resnet_18",
    "resnet_18_cls",
    "resnet_34",
    "resnet_34_cls",
    "resnet_50",
    "resnet_50_cls",
    "resnet_101",
    "resnet_101_cls",
    "resnet_152",
    "resnet_152_cls",
    "wide_resnet_50",
    "wide_resnet_50_cls",
    "wide_resnet_101",
    "wide_resnet_101_cls",
    "resnet_200",
    "resnet_200_cls",
    "resnet_269",
    "resnet_269_cls",
]
