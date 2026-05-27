"""``lucid.weights`` — pretrained-weight system for the model zoo.

A standalone package (sibling to :mod:`lucid.models`) that owns
everything about *pretrained checkpoints*: where they live, how to
fetch + verify them, the preprocessing they expect, and the tagged
variant system used to select among multiple checkpoints for one
architecture.

Design
------
Each architecture declares a :class:`WeightsEnum` (e.g.
``ResNet18Weights``) whose members are :class:`WeightEntry` manifests —
one per tagged checkpoint (``IMAGENET1K_V1``, …) plus a ``DEFAULT``
alias.  This mirrors the torchvision ``Weights`` enum so call sites are
familiar::

    import lucid.models as models
    import lucid.weights as weights

    # default tag
    model = models.resnet_18(pretrained=True)

    # explicit tag — enum or string
    model = models.resnet_18(weights=weights.vision.ResNet18Weights.IMAGENET1K_V1)
    model = models.resnet_18(pretrained="IMAGENET1K_V1")

    # discover + preprocess
    weights.list_pretrained("resnet_18")        # ['IMAGENET1K_V1']
    w = weights.get_weight("ResNet18Weights.IMAGENET1K_V1")
    x = w.transforms()(image)

The package depends only on :mod:`lucid.serialization` (for SafeTensors
loading); model factories depend on it, never the reverse, so there is
no circular import.  Crucially, :class:`WeightEntry` carries no model
config — named factories already pin their own.

Checkpoints are hosted on the Hugging Face Hub under the ``lucid-dl``
org, converted from torchvision / timm / transformers sources by the
offline ``tools/convert_weights`` pipeline.
"""

from lucid.weights import vision
from lucid.weights._base import WeightEntry, WeightsEnum
from lucid.weights._hub import download
from lucid.weights._loading import load_weight_entry, resolve_weights
from lucid.weights._registry import (
    get_weight,
    list_pretrained,
    register_weights,
    weights_for,
)
from lucid.weights._transforms import ImageClassification, Transform

__all__ = [
    # Core types
    "WeightsEnum",
    "WeightEntry",
    # Transforms
    "Transform",
    "ImageClassification",
    # Discovery
    "get_weight",
    "list_pretrained",
    "register_weights",
    "weights_for",
    # Fetch / load
    "download",
    "load_weight_entry",
    "resolve_weights",
    # Sub-packages
    "vision",
]
