"""Vision-model pretrained-weight enums.

Importing this package registers every vision architecture's weights
with the discovery registry (:func:`lucid.weights.list_pretrained` /
:func:`lucid.weights.get_weight`).
"""

from lucid.weights.vision.resnet import ResNet18Weights

__all__ = ["ResNet18Weights"]
