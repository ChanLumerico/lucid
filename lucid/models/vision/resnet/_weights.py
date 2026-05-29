"""Pretrained-weight declarations for the ResNet family.

Per-model weight enums live *with the model* (not in
:mod:`lucid.weights`).  :mod:`lucid.weights` is purely the porting
infrastructure — base classes, transforms, hub download, loading, and
the discovery registry; it knows nothing about specific architectures.
Each model family declares its checkpoints here by importing those
primitives.  Importing this module (which happens when
:mod:`lucid.models.vision.resnet` loads) registers the enums with the
discovery registry (:func:`lucid.weights.list_pretrained` /
:func:`lucid.weights.get_weight`).

These declarations are pure metadata — they pull only from
:mod:`lucid.weights`, never the reverse, so the dependency stays
one-directional (``models`` → ``weights``).

Seven paper-cited variants ship ImageNet-1k checkpoints converted from
the reference-framework model zoo: the five canonical ResNets (He et
al., CVPR 2016 — :class:`ResNet18Weights` … :class:`ResNet152Weights`)
and two Wide ResNets (Zagoruyko & Komodakis, BMVC 2016 —
:class:`WideResNet50Weights`, :class:`WideResNet101Weights`).  Every
checkpoint uses the standard ImageNet eval pipeline (224 crop / 256
resize / bilinear / ImageNet stats).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=256)


@register_weights("resnet_18")
class ResNet18Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.resnet_18_cls`.

    Ships a single ImageNet-1k checkpoint (:attr:`IMAGENET1K_V1`) — the
    canonical V1 weights distributed by the reference-framework model
    zoo, re-hosted under ``huggingface.co/lucid-dl/resnet-18`` with the
    official ``acc@1=69.758 / acc@5=89.078`` validation metrics.
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/resnet-18/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="bb7eab3083c24be6364e32f1d37844a00c5e500fa48a83a91f750a7621d152cb",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ResNet18_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            # Full provenance (training recipe URL etc.) lives in the
            # Hub config.json; kept out of runtime source per H5.
            "num_params": 11_689_512,
            "gflops": 1.814,
            "file_size_mb": 44.7,
            "metrics": {"ImageNet-1k": {"acc@1": 69.758, "acc@5": 89.078}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("resnet_34_cls")
class ResNet34Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.resnet_34_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/resnet-34/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="5d11396194da73066c1007086823da38447031d9dd260b8a2878e834245d3271",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ResNet34_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 21_797_672,
            "gflops": 3.664,
            "metrics": {"ImageNet-1k": {"acc@1": 73.314, "acc@5": 91.420}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("resnet_50_cls")
class ResNet50Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.resnet_50_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/resnet-50/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="c634bc6bdc82cf41c7b630c3ab01f4d6896c3b5bbe39f987e4c2a80c31360777",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ResNet50_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 25_557_032,
            "gflops": 4.089,
            "metrics": {"ImageNet-1k": {"acc@1": 76.130, "acc@5": 92.862}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("resnet_101_cls")
class ResNet101Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.resnet_101_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/resnet-101/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="531456a9dac7c8b836a92210f7db9e65d1f9c1e0530a70721249622577b3c02b",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ResNet101_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 44_549_160,
            "gflops": 7.801,
            "metrics": {"ImageNet-1k": {"acc@1": 77.374, "acc@5": 93.546}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("resnet_152_cls")
class ResNet152Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.resnet_152_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/resnet-152/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="52dde167f0e0292cea56165c3a7788020a69f3507aeb0e766ad0fe7dcc0f72f4",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ResNet152_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 60_192_808,
            "gflops": 11.514,
            "metrics": {"ImageNet-1k": {"acc@1": 78.312, "acc@5": 94.046}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("wide_resnet_50_cls")
class WideResNet50Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.wide_resnet_50_cls`.

    Wide ResNet-50-2 (Zagoruyko & Komodakis, "Wide Residual Networks",
    BMVC 2016).
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/wide-resnet-50-2/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="e3448746102221b7d2ab734ba244b5312b444b26324484b97e0eb30ff364bd46",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/Wide_ResNet50_2_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 68_883_240,
            "gflops": 11.398,
            "metrics": {"ImageNet-1k": {"acc@1": 78.468, "acc@5": 94.086}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("wide_resnet_101_cls")
class WideResNet101Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.wide_resnet_101_cls`.

    Wide ResNet-101-2 (Zagoruyko & Komodakis, "Wide Residual Networks",
    BMVC 2016).
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/wide-resnet-101-2/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="935b3ddae9b1329af8802d7346a3a5fcc2faef45415e620563076d4123409c4f",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/Wide_ResNet101_2_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 126_886_696,
            "gflops": 22.753,
            "metrics": {"ImageNet-1k": {"acc@1": 78.848, "acc@5": 94.284}},
        },
    )
    DEFAULT = IMAGENET1K_V1
