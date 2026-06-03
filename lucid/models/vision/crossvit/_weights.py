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

_PRESET = ImageClassification(crop_size=240, resize_size=274, interpolation="bicubic")


@register_weights("crossvit_tiny_cls")
class CrossViTTinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.crossvit_tiny_cls`.

    Chen et al. ICCV 2021 dual-branch ViT (7.0 M params, top-1 72.6%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 72.6%), sourced from
        ``timm/crossvit_tiny_240.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Chen, Fan, Panda, *"CrossViT: Cross-Attention Multi-Scale
    Vision Transformer for Image Classification"*, ICCV 2021
    (arXiv:2103.14899).

    Examples
    --------
    >>> from lucid.models import crossvit_tiny_cls
    >>> model = crossvit_tiny_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-tiny/resolve/main/IN1K/model.safetensors",
        sha256="f0213ad20473517f15f176f5f086962dcee31c60b910f8a113a8dba5e324645d",
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
    r"""Pretrained weights for :func:`lucid.models.crossvit_small_cls`.

    Chen et al. ICCV 2021 dual-branch ViT (26.9 M params, top-1 81.0%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 81.0%), sourced from
        ``timm/crossvit_small_240.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Chen, Fan, Panda, *"CrossViT: Cross-Attention Multi-Scale
    Vision Transformer for Image Classification"*, ICCV 2021
    (arXiv:2103.14899).

    Examples
    --------
    >>> from lucid.models import crossvit_small_cls
    >>> model = crossvit_small_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-small/resolve/main/IN1K/model.safetensors",
        sha256="3b5a05f3a82a44eb6313368b9244c8a526ea5fd8cf0f13b7dd4d81ade476ca8f",
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
    r"""Pretrained weights for :func:`lucid.models.crossvit_base_cls`.

    Chen et al. ICCV 2021 dual-branch ViT (105 M params, top-1 82.2%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 82.2%), sourced from
        ``timm/crossvit_base_240.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Chen, Fan, Panda, *"CrossViT: Cross-Attention Multi-Scale
    Vision Transformer for Image Classification"*, ICCV 2021
    (arXiv:2103.14899).

    Examples
    --------
    >>> from lucid.models import crossvit_base_cls
    >>> model = crossvit_base_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-base/resolve/main/IN1K/model.safetensors",
        sha256="780bd01e7f83e79d4bd0bbec7a2527ee7771b52e222d3612849705718f26bc1f",
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
    r"""Pretrained weights for :func:`lucid.models.crossvit_9_cls`.

    9-block CrossViT variant (Chen et al. ICCV 2021; 8.6 M params, top-1 73.9%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 73.9%), sourced from
        ``timm/crossvit_9_240.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Chen, Fan, Panda, *"CrossViT: Cross-Attention Multi-Scale
    Vision Transformer for Image Classification"*, ICCV 2021
    (arXiv:2103.14899).

    Examples
    --------
    >>> from lucid.models import crossvit_9_cls
    >>> model = crossvit_9_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-9/resolve/main/IN1K/model.safetensors",
        sha256="fc4a65ad01b32d198f6dc106b3e3b3c012f8114143c2585e4eef0ae28986e5da",
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
    r"""Pretrained weights for :func:`lucid.models.crossvit_15_cls`.

    15-block CrossViT variant (Chen et al. ICCV 2021; 27.5 M params, top-1 81.5%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 81.5%), sourced from
        ``timm/crossvit_15_240.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Chen, Fan, Panda, *"CrossViT: Cross-Attention Multi-Scale
    Vision Transformer for Image Classification"*, ICCV 2021
    (arXiv:2103.14899).

    Examples
    --------
    >>> from lucid.models import crossvit_15_cls
    >>> model = crossvit_15_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-15/resolve/main/IN1K/model.safetensors",
        sha256="e3dc9964773551bfec1031e781a72ac7aa13cf6830749b6466ec0d0c0dedc100",
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
    r"""Pretrained weights for :func:`lucid.models.crossvit_18_cls`.

    18-block CrossViT variant (Chen et al. ICCV 2021; 43.3 M params, top-1 82.5%).

    Attributes
    ----------
    IN1K : WeightEntry
        ImageNet-1k checkpoint (top-1 82.5%), sourced from
        ``timm/crossvit_18_240.in1k``.
    DEFAULT : WeightEntry
        Alias for :attr:`IN1K`.

    Notes
    -----
    Reference: Chen, Fan, Panda, *"CrossViT: Cross-Attention Multi-Scale
    Vision Transformer for Image Classification"*, ICCV 2021
    (arXiv:2103.14899).

    Examples
    --------
    >>> from lucid.models import crossvit_18_cls
    >>> model = crossvit_18_cls(pretrained=True).eval()
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/crossvit-18/resolve/main/IN1K/model.safetensors",
        sha256="7238e4b62ce35b543018f0f5b3ba9ad12b48ef19d125ed6065e303c4a2479419",
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
