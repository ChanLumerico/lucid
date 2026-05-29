"""Pretrained-weight declarations for the SK-ResNet family.

Two paper-cited variants (Li et al., CVPR 2019) ship ImageNet-1k
checkpoints converted from the ``timm`` ``skresnet18.ra_in1k`` /
``skresnet34.ra_in1k`` weights (Wightman's RandAugment recipe).  The
``ra_in1k`` eval pipeline uses a 224 centre crop at
``crop_pct = 0.875`` (→ 256 resize) with **bicubic** interpolation
and ImageNet mean/std.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# timm ``ra_in1k`` preset: 224 crop / 0.875 crop_pct → 256 resize, bicubic.
_PRESET = ImageClassification(crop_size=224, resize_size=256, interpolation="bicubic")


@register_weights("sk_resnet_18_cls")
class SKResNet18Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.sk_resnet_18_cls`.

    Single ImageNet-1k checkpoint (:attr:`RA_IN1K`) converted from
    ``timm/skresnet18.ra_in1k`` — Wightman's RandAugment recipe, hosted
    under ``huggingface.co/lucid-dl/sk-resnet-18`` with the official
    ``acc@1 = 73.020 / acc@5 = 91.172`` validation metrics.
    """

    RA_IN1K = WeightEntry(
        url=f"{HUB_BASE}/sk-resnet-18/resolve/main/RA_IN1K/model.safetensors",
        sha256="eee529647dfe98f5397efd0764d17ab7319e730f1f192dd73dd9f4a7763a85dd",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "RA_IN1K",
            "source": "timm/skresnet18.ra_in1k",
            "license": "apache-2.0",
            "num_params": 11_958_056,
            "metrics": {"ImageNet-1k": {"acc@1": 73.020, "acc@5": 91.172}},
        },
    )
    DEFAULT = RA_IN1K


@register_weights("sk_resnet_34_cls")
class SKResNet34Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.sk_resnet_34_cls`.

    Single ImageNet-1k checkpoint (:attr:`RA_IN1K`) converted from
    ``timm/skresnet34.ra_in1k`` — Wightman's RandAugment recipe, hosted
    under ``huggingface.co/lucid-dl/sk-resnet-34`` with the official
    ``acc@1 = 76.956 / acc@5 = 93.320`` validation metrics.
    """

    RA_IN1K = WeightEntry(
        url=f"{HUB_BASE}/sk-resnet-34/resolve/main/RA_IN1K/model.safetensors",
        sha256="d36f565612c3929a4a64d5449c30b3d8f3e2cef9ec4fd4765229883f19edc3eb",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "RA_IN1K",
            "source": "timm/skresnet34.ra_in1k",
            "license": "apache-2.0",
            "num_params": 22_282_376,
            "metrics": {"ImageNet-1k": {"acc@1": 76.956, "acc@5": 93.320}},
        },
    )
    DEFAULT = RA_IN1K
