"""Pretrained-weight declarations for the CrossViT family.

Six paper-cited variants (tiny / small / base / 9 / 15 / 18); every
checkpoint is sourced from timm's ``crossvit_<variant>_240.in1k`` tag
(Chen et al., ICCV 2021 — ImageNet-1k training only; no ImageNet-22k
pretraining for the canonical CrossViT line).

Preset: ``crop=240``, ``resize=274`` (timm's ``crop_pct=0.875``),
``bicubic`` interpolation, ImageNet mean/std — identical for every
variant.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(
    crop_size=240, resize_size=274, interpolation="bicubic"
)


@register_weights("crossvit_tiny_cls")
class CrossViTTinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.crossvit_tiny_cls`."""

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-tiny/resolve/main/IN1K/model.safetensors",
        sha256="__PENDING_UPLOAD__",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/crossvit_tiny_240.in1k",
            "license": "apache-2.0",
            "num_params": 7_010_400,
            "metrics": {"ImageNet-1k": {"acc@1": 72.6}},
        },
    )
    DEFAULT = IN1K


@register_weights("crossvit_small_cls")
class CrossViTSmallWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.crossvit_small_cls`."""

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-small/resolve/main/IN1K/model.safetensors",
        sha256="__PENDING_UPLOAD__",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/crossvit_small_240.in1k",
            "license": "apache-2.0",
            "num_params": 26_855_192,
            "metrics": {"ImageNet-1k": {"acc@1": 81.0}},
        },
    )
    DEFAULT = IN1K


@register_weights("crossvit_base_cls")
class CrossViTBaseWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.crossvit_base_cls`."""

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-base/resolve/main/IN1K/model.safetensors",
        sha256="__PENDING_UPLOAD__",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/crossvit_base_240.in1k",
            "license": "apache-2.0",
            "num_params": 105_028_344,
            "metrics": {"ImageNet-1k": {"acc@1": 82.2}},
        },
    )
    DEFAULT = IN1K


@register_weights("crossvit_9_cls")
class CrossViT9Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.crossvit_9_cls`."""

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-9/resolve/main/IN1K/model.safetensors",
        sha256="__PENDING_UPLOAD__",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/crossvit_9_240.in1k",
            "license": "apache-2.0",
            "num_params": 8_553_600,
            "metrics": {"ImageNet-1k": {"acc@1": 73.9}},
        },
    )
    DEFAULT = IN1K


@register_weights("crossvit_15_cls")
class CrossViT15Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.crossvit_15_cls`."""

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-15/resolve/main/IN1K/model.safetensors",
        sha256="__PENDING_UPLOAD__",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/crossvit_15_240.in1k",
            "license": "apache-2.0",
            "num_params": 27_528_120,
            "metrics": {"ImageNet-1k": {"acc@1": 81.5}},
        },
    )
    DEFAULT = IN1K


@register_weights("crossvit_18_cls")
class CrossViT18Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.crossvit_18_cls`."""

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-18/resolve/main/IN1K/model.safetensors",
        sha256="__PENDING_UPLOAD__",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/crossvit_18_240.in1k",
            "license": "apache-2.0",
            "num_params": 43_270_648,
            "metrics": {"ImageNet-1k": {"acc@1": 82.5}},
        },
    )
    DEFAULT = IN1K
