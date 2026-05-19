"""Vision Transformer (ViT) model family — Dosovitskiy et al., 2020.

This sub-package exposes the canonical Vision Transformer architecture
introduced in *"An Image is Worth 16x16 Words: Transformers for Image
Recognition at Scale"* (ICLR 2021, `arXiv:2010.11929
<https://arxiv.org/abs/2010.11929>`_).

The architecture splits an image into non-overlapping square patches,
linearly projects each patch into a :math:`D`-dimensional embedding,
prepends a learnable ``[CLS]`` token, adds a learnable positional
embedding, and processes the resulting sequence with a stack of pre-norm
transformer encoder blocks.  The final CLS-token representation is used
as the global image feature.

Exports
-------
Configuration:

* :class:`ViTConfig` — immutable dataclass describing any ViT variant.

Modules:

* :class:`ViT` — feature-extractor backbone (returns ``(B, dim)`` CLS).
* :class:`ViTForImageClassification` — backbone + linear head producing
  classification logits.

Backbone factories:

* :func:`vit_base_16`, :func:`vit_base_32`
* :func:`vit_large_16`, :func:`vit_large_32`
* :func:`vit_huge_14`

Classifier factories (CLS-token linear head):

* :func:`vit_base_16_cls`, :func:`vit_base_32_cls`
* :func:`vit_large_16_cls`, :func:`vit_large_32_cls`
* :func:`vit_huge_14_cls`
"""

from lucid.models.vision.vit._config import ViTConfig
from lucid.models.vision.vit._model import ViT, ViTForImageClassification
from lucid.models.vision.vit._pretrained import (
    vit_base_16,
    vit_base_16_cls,
    vit_base_32,
    vit_base_32_cls,
    vit_large_16,
    vit_large_16_cls,
    vit_large_32,
    vit_large_32_cls,
    vit_huge_14,
    vit_huge_14_cls,
)

__all__ = [
    "ViTConfig",
    "ViT",
    "ViTForImageClassification",
    "vit_base_16",
    "vit_base_16_cls",
    "vit_base_32",
    "vit_base_32_cls",
    "vit_large_16",
    "vit_large_16_cls",
    "vit_large_32",
    "vit_large_32_cls",
    "vit_huge_14",
    "vit_huge_14_cls",
]
