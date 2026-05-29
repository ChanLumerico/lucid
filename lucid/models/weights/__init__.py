"""``lucid.models.weights`` — per-family pretrained-weight enums.

Discovery namespace for every model family's ``<Variant>Weights`` enum.
Each enum is *declared* inside its model package
(``lucid/models/vision/<family>/_weights.py``) so the declaration sits
right next to the architecture it targets; this module then re-exports
all of them under a single short import path so users don't have to
remember which sub-package each family lives in::

    from lucid.models.weights import AlexNetWeights, ResNet18Weights
    from lucid.models import alexnet_cls
    m = alexnet_cls(weights=AlexNetWeights.IMAGENET1K_V1)

Deliberately separate from :mod:`lucid.weights` — that package hosts the
*infrastructure* (``WeightEntry``, ``WeightsEnum``, ``HUB_BASE``,
``register_weights``, hub download, loading); this package only re-exports
*concrete* enums.  And deliberately separate from the top-level
:mod:`lucid.models` namespace — that namespace already carries every
model class + factory, so piling on tens of additional ``*Weights``
symbols would bury discovery.  All ``<Variant>Weights`` enums are
reachable here and *only* here:

* ✅ ``from lucid.models.weights import AlexNetWeights``
* ❌ ``from lucid.models import AlexNetWeights`` (not exported)

The dependency direction is one-way (``models.weights`` → per-family
``_weights.py`` → :mod:`lucid.weights`); the per-family declarations
register themselves with :mod:`lucid.weights` for the discovery
registry on import, so importing :mod:`lucid.models.weights` (or any
member of it) also populates :func:`lucid.weights.list_pretrained`.
"""

# 2012 — AlexNet (Krizhevsky 2014 single-stream OWT after NIPS 2012)
from lucid.models.vision.alexnet._weights import AlexNetWeights

# 2016 — DenseNet (Huang et al.)
from lucid.models.vision.densenet._weights import (
    DenseNet121Weights,
    DenseNet161Weights,
    DenseNet169Weights,
    DenseNet201Weights,
)

# 2014 — VGG (Simonyan & Zisserman)
from lucid.models.vision.vgg._weights import (
    VGG11Weights,
    VGG13Weights,
    VGG16Weights,
    VGG19Weights,
    VGG11BNWeights,
    VGG13BNWeights,
    VGG16BNWeights,
    VGG19BNWeights,
)

# 2015 — ResNet (He et al.) — full variant set + Wide ResNet
from lucid.models.vision.resnet._weights import (
    ResNet18Weights,
    ResNet34Weights,
    ResNet50Weights,
    ResNet101Weights,
    ResNet152Weights,
    WideResNet50Weights,
    WideResNet101Weights,
)

# 2017 — ResNeXt (Xie et al.)
from lucid.models.vision.resnext._weights import (
    ResNeXt50_32x4dWeights,
    ResNeXt101_32x4dWeights,
    ResNeXt101_32x8dWeights,
)

# 2015 — Inception v3 (Szegedy et al.)
from lucid.models.vision.inception._weights import InceptionV3Weights

# 2023 — InceptionNeXt (Yu et al.)
from lucid.models.vision.inception_next._weights import (
    InceptionNeXtTinyWeights,
    InceptionNeXtSmallWeights,
    InceptionNeXtBaseWeights,
)

# 2016 — Inception-ResNet v2 (Szegedy et al.)
from lucid.models.vision.inception_resnet._weights import InceptionResNetV2Weights

# 2017 — Xception (Chollet)
from lucid.models.vision.xception._weights import XceptionWeights

# 2018 — MobileNetV2 (Sandler et al.)
from lucid.models.vision.mobilenet_v2._weights import MobileNetV2Weights

# 2019 — MobileNetV3 (Howard et al.)
from lucid.models.vision.mobilenet_v3._weights import (
    MobileNetV3LargeWeights,
    MobileNetV3SmallWeights,
)

# 2018 — SENet (Hu et al.)
from lucid.models.vision.senet._weights import (
    SEResNet18Weights,
    SEResNet34Weights,
    SEResNet50Weights,
    SEResNet101Weights,
    SEResNet152Weights,
)

# 2019 — EfficientNet (Tan & Le)
from lucid.models.vision.efficientnet._weights import (
    EfficientNetB0Weights,
    EfficientNetB1Weights,
    EfficientNetB2Weights,
    EfficientNetB3Weights,
    EfficientNetB4Weights,
    EfficientNetB5Weights,
    EfficientNetB6Weights,
    EfficientNetB7Weights,
)

# 2019 — SKNet (Li et al.)
from lucid.models.vision.sknet._weights import (
    SKResNet18Weights,
    SKResNet34Weights,
)

# 2019 — CSPNet (Wang et al.)
from lucid.models.vision.cspnet._weights import (
    CSPDarknet53Weights,
    CSPResNet50Weights,
    CSPResNeXt50Weights,
)

# 2021 — CrossViT (Chen et al.)
from lucid.models.vision.crossvit._weights import (
    CrossViT9Weights,
    CrossViT15Weights,
    CrossViT18Weights,
    CrossViTBaseWeights,
    CrossViTSmallWeights,
    CrossViTTinyWeights,
)

# 2021 — CvT (Wu et al.)
from lucid.models.vision.cvt._weights import (
    CvT13Weights,
    CvT21Weights,
    CvTW24Weights,
)

# 2022 — ConvNeXt (Liu et al.)
from lucid.models.vision.convnext._weights import (
    ConvNeXtTinyWeights,
    ConvNeXtSmallWeights,
    ConvNeXtBaseWeights,
    ConvNeXtLargeWeights,
    ConvNeXtXLargeWeights,
)

# 2022 — PVTv2 (Wang et al.)
from lucid.models.vision.pvt._weights import (
    PVTv2B0Weights,
    PVTv2B1Weights,
    PVTv2B2Weights,
    PVTv2B3Weights,
    PVTv2B4Weights,
    PVTv2B5Weights,
)

# 2014 — GoogLeNet / Inception v1 (Szegedy et al.)
from lucid.models.vision.googlenet._weights import GoogLeNetWeights

# 2020 — DETR object detection (Carion et al.) — COCO
from lucid.models.vision.detr._weights import (
    DETRResNet50Weights,
    DETRResNet101Weights,
)

# 2017 — MobileNet v1 (Howard et al.)
from lucid.models.vision.mobilenet._weights import MobileNetV1Weights

# 2022 — EfficientFormer (Li et al.)
from lucid.models.vision.efficientformer._weights import (
    EfficientFormerL1Weights,
    EfficientFormerL3Weights,
    EfficientFormerL7Weights,
)

# 2020 — Vision Transformer (Dosovitskiy et al.)
from lucid.models.vision.vit._weights import (
    ViTBase16Weights,
    ViTBase32Weights,
    ViTLarge16Weights,
    ViTLarge32Weights,
)

# 2020 — ResNeSt (Zhang et al.)
from lucid.models.vision.resnest._weights import (
    ResNeSt50Weights,
    ResNeSt101Weights,
    ResNeSt200Weights,
    ResNeSt269Weights,
)

# 2021 — Swin Transformer (Liu et al.)
from lucid.models.vision.swin._weights import (
    SwinTinyWeights,
    SwinSmallWeights,
    SwinBaseWeights,
    SwinLargeWeights,
)

# 2022 — MaxViT (Tu et al.)
from lucid.models.vision.maxvit._weights import (
    MaxViTTinyWeights,
    MaxViTSmallWeights,
    MaxViTBaseWeights,
    MaxViTLargeWeights,
)

__all__ = [
    # ── Vision (2012) AlexNet ─────────────────────────────────────────
    "AlexNetWeights",
    # ── Vision (2014) VGG ─────────────────────────────────────────────
    "VGG11Weights",
    "VGG13Weights",
    "VGG16Weights",
    "VGG19Weights",
    "VGG11BNWeights",
    "VGG13BNWeights",
    "VGG16BNWeights",
    "VGG19BNWeights",
    # ── Vision (2015) ResNet ──────────────────────────────────────────
    "ResNet18Weights",
    "ResNet34Weights",
    "ResNet50Weights",
    "ResNet101Weights",
    "ResNet152Weights",
    "WideResNet50Weights",
    "WideResNet101Weights",
    # ── Vision (2017) ResNeXt ─────────────────────────────────────────
    "ResNeXt50_32x4dWeights",
    "ResNeXt101_32x4dWeights",
    "ResNeXt101_32x8dWeights",
    # ── Vision (2015) Inception v3 ────────────────────────────────────
    "InceptionV3Weights",
    # ── Vision (2023) InceptionNeXt ───────────────────────────────────
    "InceptionNeXtTinyWeights",
    "InceptionNeXtSmallWeights",
    "InceptionNeXtBaseWeights",
    # ── Vision (2016) DenseNet ────────────────────────────────────────
    "DenseNet121Weights",
    "DenseNet161Weights",
    "DenseNet169Weights",
    "DenseNet201Weights",
    # ── Vision (2016) Inception-ResNet v2 ─────────────────────────────
    "InceptionResNetV2Weights",
    # ── Vision (2017) Xception ────────────────────────────────────────
    "XceptionWeights",
    # ── Vision (2018) MobileNetV2 ─────────────────────────────────────
    "MobileNetV2Weights",
    # ── Vision (2019) MobileNetV3 ─────────────────────────────────────
    "MobileNetV3LargeWeights",
    "MobileNetV3SmallWeights",
    # ── Vision (2018) SENet ───────────────────────────────────────────
    "SEResNet18Weights",
    "SEResNet34Weights",
    "SEResNet50Weights",
    "SEResNet101Weights",
    "SEResNet152Weights",
    # ── Vision (2019) EfficientNet ────────────────────────────────────
    "EfficientNetB0Weights",
    "EfficientNetB1Weights",
    "EfficientNetB2Weights",
    "EfficientNetB3Weights",
    "EfficientNetB4Weights",
    "EfficientNetB5Weights",
    "EfficientNetB6Weights",
    "EfficientNetB7Weights",
    # ── Vision (2019) SKNet ───────────────────────────────────────────
    "SKResNet18Weights",
    "SKResNet34Weights",
    # ── Vision (2019) CSPNet ──────────────────────────────────────────
    "CSPResNet50Weights",
    "CSPResNeXt50Weights",
    "CSPDarknet53Weights",
    # ── Vision (2021) CrossViT ────────────────────────────────────────
    "CrossViTTinyWeights",
    "CrossViTSmallWeights",
    "CrossViTBaseWeights",
    "CrossViT9Weights",
    "CrossViT15Weights",
    "CrossViT18Weights",
    # ── Vision (2021) CvT ─────────────────────────────────────────────
    "CvT13Weights",
    "CvT21Weights",
    "CvTW24Weights",
    # ── Vision (2022) ConvNeXt ────────────────────────────────────────
    "ConvNeXtTinyWeights",
    "ConvNeXtSmallWeights",
    "ConvNeXtBaseWeights",
    "ConvNeXtLargeWeights",
    "ConvNeXtXLargeWeights",
    # ── Vision (2022) PVTv2 ───────────────────────────────────────────
    "PVTv2B0Weights",
    "PVTv2B1Weights",
    "PVTv2B2Weights",
    "PVTv2B3Weights",
    "PVTv2B4Weights",
    "PVTv2B5Weights",
    # ── Vision (2014) GoogLeNet ───────────────────────────────────────
    "GoogLeNetWeights",
    # ── Vision (2020) DETR (COCO object detection) ────────────────────
    "DETRResNet50Weights",
    "DETRResNet101Weights",
    # ── Vision (2017) MobileNet v1 ────────────────────────────────────
    "MobileNetV1Weights",
    # ── Vision (2022) EfficientFormer ─────────────────────────────────
    "EfficientFormerL1Weights",
    "EfficientFormerL3Weights",
    "EfficientFormerL7Weights",
    # ── Vision (2020) Vision Transformer ──────────────────────────────
    "ViTBase16Weights",
    "ViTBase32Weights",
    "ViTLarge16Weights",
    "ViTLarge32Weights",
    # ── Vision (2020) ResNeSt ─────────────────────────────────────────
    "ResNeSt50Weights",
    "ResNeSt101Weights",
    "ResNeSt200Weights",
    "ResNeSt269Weights",
    # ── Vision (2021) Swin Transformer ────────────────────────────────
    "SwinTinyWeights",
    "SwinSmallWeights",
    "SwinBaseWeights",
    "SwinLargeWeights",
    # ── Vision (2022) MaxViT ──────────────────────────────────────────
    "MaxViTTinyWeights",
    "MaxViTSmallWeights",
    "MaxViTBaseWeights",
    "MaxViTLargeWeights",
]
