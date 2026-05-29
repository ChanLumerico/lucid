"""Pretrained-weight declarations for the VGG family.

Eight paper-cited variants (Simonyan & Zisserman, ICLR 2015 —
configurations A/B/D/E ≡ VGG-11/13/16/19, each with and without
BatchNorm) — all converted from torchvision's
``VGG{11,13,16,19}[_BN]_Weights.IMAGENET1K_V1`` tag.  The BatchNorm
variants are not in the original paper; they were added in the
torchvision / timm reimplementations and converge faster with higher
final accuracy.

Every checkpoint uses the standard ImageNet eval pipeline (224 crop /
256 resize / bilinear / ImageNet stats).  Reported ``acc@1`` are the
official torchvision validation top-1 figures (the paper itself only
tabulates top-5 error).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=256)


@register_weights("vgg_11_cls")
class VGG11Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_11_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-11/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="59618a0dc0607270fa29dc0d878951b4163813f13bb2c9124242026510a6225a",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG11_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 132_863_336,
            "gflops": 7.609,
            "metrics": {"ImageNet-1k": {"acc@1": 69.020, "acc@5": 88.628}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vgg_13_cls")
class VGG13Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_13_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-13/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="9004cbee1cf4f8cfd98b636353a6a10ee528e8b30e802f381915cfc9181a8903",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG13_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 133_047_848,
            "gflops": 11.308,
            "metrics": {"ImageNet-1k": {"acc@1": 69.928, "acc@5": 89.246}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vgg_16_cls")
class VGG16Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_16_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-16/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="f8518a067555b025a2da17f1fa106bb857dd6004a3a01dfcc77fc28b836898a6",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG16_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 138_357_544,
            "gflops": 15.470,
            "metrics": {"ImageNet-1k": {"acc@1": 71.592, "acc@5": 90.382}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vgg_19_cls")
class VGG19Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_19_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-19/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="91e37c12f5d3c2765b3703e171249b329e5171ecdd1d340fb7ce6f5adf738127",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG19_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 143_667_240,
            "gflops": 19.632,
            "metrics": {"ImageNet-1k": {"acc@1": 72.376, "acc@5": 90.876}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vgg_11_bn_cls")
class VGG11BNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_11_bn_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-11-bn/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="4ec413e0c4e21dd660c463e3bf9f157556aa162121906f07a9ad4c4d902c0ac4",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG11_BN_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 132_868_840,
            "gflops": 7.609,
            "metrics": {"ImageNet-1k": {"acc@1": 70.370, "acc@5": 89.810}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vgg_13_bn_cls")
class VGG13BNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_13_bn_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-13-bn/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="e288832644ed8355ccdb24bd2f4b286f21b21daf3804aa6633adc07572839ae9",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG13_BN_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 133_053_736,
            "gflops": 11.308,
            "metrics": {"ImageNet-1k": {"acc@1": 71.586, "acc@5": 90.374}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vgg_16_bn_cls")
class VGG16BNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_16_bn_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-16-bn/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="d8d00db5676660816d4c0eabf4f203c38e5eadb16daab58c1475f949f608aa09",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG16_BN_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 138_365_992,
            "gflops": 15.470,
            "metrics": {"ImageNet-1k": {"acc@1": 73.360, "acc@5": 91.516}},
        },
    )
    DEFAULT = IMAGENET1K_V1


@register_weights("vgg_19_bn_cls")
class VGG19BNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.vgg_19_bn_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/vgg-19-bn/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="18f776e34456582dc812e5dd2080ba582a707707982b91d7119e405786bb41c9",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/VGG19_BN_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 143_678_248,
            "gflops": 19.632,
            "metrics": {"ImageNet-1k": {"acc@1": 74.218, "acc@5": 91.842}},
        },
    )
    DEFAULT = IMAGENET1K_V1
