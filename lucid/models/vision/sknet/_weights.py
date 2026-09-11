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

    Examples
    --------
    >>> from lucid.models.vision.sknet._weights import SKResNet18Weights
    >>> list(SKResNet18Weights.__members__)
    ['RA_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> SKResNet18Weights.DEFAULT is SKResNet18Weights.RA_IN1K
    True
    >>> SKResNet18Weights.RA_IN1K.num_classes
    1000
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
        # timm's skresnet18/34 carry no Eq. (4) floor, so the two narrow
        # stages are 16 wide and not the paper's 32.  A model built at 32
        # fails on 24 shapes; declaring it says so before the download.
        requires_config={"min_attn_channels": 16},
    )
    DEFAULT = RA_IN1K


@register_weights("sk_resnet_34_cls")
class SKResNet34Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.sk_resnet_34_cls`.

    Single ImageNet-1k checkpoint (:attr:`RA_IN1K`) converted from
    ``timm/skresnet34.ra_in1k`` — Wightman's RandAugment recipe, hosted
    under ``huggingface.co/lucid-dl/sk-resnet-34`` with the official
    ``acc@1 = 76.956 / acc@5 = 93.320`` validation metrics.

    Examples
    --------
    >>> from lucid.models.vision.sknet._weights import SKResNet34Weights
    >>> list(SKResNet34Weights.__members__)
    ['RA_IN1K', 'DEFAULT']

    ``DEFAULT`` is an alias rather than a fourth entry, so a bare
    ``pretrained=True`` and the tag it resolves to cannot drift apart.

    >>> SKResNet34Weights.DEFAULT is SKResNet34Weights.RA_IN1K
    True
    >>> SKResNet34Weights.RA_IN1K.num_classes
    1000
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
        # timm's skresnet18/34 carry no Eq. (4) floor, so the two narrow
        # stages are 16 wide and not the paper's 32.  A model built at 32
        # fails on 24 shapes; declaring it says so before the download.
        requires_config={"min_attn_channels": 16},
    )
    DEFAULT = RA_IN1K
