"""Pretrained-weight declarations for the DenseNet family.

Four paper-cited variants (Huang et al., CVPR 2017) — all converted
from torchvision's ``DenseNet*_Weights.IMAGENET1K_V1`` tag.  Preset is
the standard ImageNet eval pipeline (224 crop / 256 resize / bilinear /
ImageNet stats) for every variant.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=256)


@register_weights("densenet_121_cls")
class DenseNet121Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.densenet_121_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/densenet-121/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="6d1e228bc99a03b0d9625d952c4f705487a90618a7c51b0eb3473164042f05d6",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/DenseNet121_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 7_978_856,
            "metrics": {"ImageNet-1k": {"acc@1": 74.434, "acc@5": 91.972}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("densenet_161_cls")
class DenseNet161Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.densenet_161_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/densenet-161/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="6937c3d766e1b7bc012517f586b7a33b6ebe6f3f3e9e1f12cfa7237a28b19297",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/DenseNet161_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 28_681_000,
            "metrics": {"ImageNet-1k": {"acc@1": 77.138, "acc@5": 93.560}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("densenet_169_cls")
class DenseNet169Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.densenet_169_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/densenet-169/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="e393b713dd69a4ab545b706bba57aba2fa0d51a2542bdcaab7ea9a502be24f15",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/DenseNet169_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 14_149_480,
            "metrics": {"ImageNet-1k": {"acc@1": 75.600, "acc@5": 92.806}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("densenet_201_cls")
class DenseNet201Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.densenet_201_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/densenet-201/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="d3afc346c153ac4002ca915b797856998d720697e57f829085b8744810f02aa6",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/DenseNet201_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 20_013_928,
            "metrics": {"ImageNet-1k": {"acc@1": 76.896, "acc@5": 93.370}},
        },
    )
    DEFAULT = IMAGENET1K_V1
