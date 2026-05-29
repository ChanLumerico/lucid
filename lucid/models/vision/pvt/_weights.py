"""Pretrained-weight declarations for the PVT v2 family.

Five paper-cited variants (Wang et al., CVMJ 2022) ship ImageNet-1k
checkpoints converted from timm's ``pvt_v2_b{0,2,3,4,5}.in1k`` model
zoo: :class:`PVTv2B0Weights`, :class:`PVTv2B2Weights`,
:class:`PVTv2B3Weights`, :class:`PVTv2B4Weights`, :class:`PVTv2B5Weights`.

PVT v2-B1 is intentionally not shipped: Lucid's B1 config currently uses
``depths=(2, 2, 4, 2)`` (~17.3M params) whereas the paper / timm
``pvt_v2_b1`` is ``depths=(2, 2, 2, 2)`` (~14.0M), so the upstream
checkpoint cannot load into Lucid's B1 without a config change.

Every checkpoint uses the timm PVT v2 eval pipeline: 224 crop / 249
resize (``crop_pct = 0.9``) / bicubic interpolation / ImageNet stats.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=249, interpolation="bicubic")


@register_weights("pvt_v2_b0_cls")
class PVTv2B0Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.pvt_v2_b0_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``pvt_v2_b0.in1k`` (Wang et al., 2022; ~3.7M params, 70.5% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/pvt-v2-b0/resolve/main/IN1K/model.safetensors",
        sha256="a511e23835b9b510bc02436d27c49dc306c97dd3b627fdc70fcbac14d5b2c771",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/pvt_v2_b0.in1k",
            "license": "apache-2.0",
            "num_params": 3_666_760,
            "metrics": {"ImageNet-1k": {"acc@1": 70.5}},
        },
    )
    DEFAULT = IN1K


@register_weights("pvt_v2_b2_cls")
class PVTv2B2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.pvt_v2_b2_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``pvt_v2_b2.in1k`` (Wang et al., 2022; ~25.4M params, 82.0% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/pvt-v2-b2/resolve/main/IN1K/model.safetensors",
        sha256="00fc4d873e5cfc8e57bb9471ffa1a2b142c0bb465bfc0a02dd347ad1812a4b71",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/pvt_v2_b2.in1k",
            "license": "apache-2.0",
            "num_params": 25_362_856,
            "metrics": {"ImageNet-1k": {"acc@1": 82.0}},
        },
    )
    DEFAULT = IN1K


@register_weights("pvt_v2_b3_cls")
class PVTv2B3Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.pvt_v2_b3_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``pvt_v2_b3.in1k`` (Wang et al., 2022; ~45.2M params, 83.1% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/pvt-v2-b3/resolve/main/IN1K/model.safetensors",
        sha256="9aebf5a6685261a41a72495724752e16a4b8d02c6e7a0977adc77cd7921dfb74",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/pvt_v2_b3.in1k",
            "license": "apache-2.0",
            "num_params": 45_238_696,
            "metrics": {"ImageNet-1k": {"acc@1": 83.1}},
        },
    )
    DEFAULT = IN1K


@register_weights("pvt_v2_b4_cls")
class PVTv2B4Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.pvt_v2_b4_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``pvt_v2_b4.in1k`` (Wang et al., 2022; ~62.6M params, 83.6% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/pvt-v2-b4/resolve/main/IN1K/model.safetensors",
        sha256="2e411209ef6a4807c09cf2decca4711236ed888dbe211645a83b0c2bf26220be",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/pvt_v2_b4.in1k",
            "license": "apache-2.0",
            "num_params": 62_556_072,
            "metrics": {"ImageNet-1k": {"acc@1": 83.6}},
        },
    )
    DEFAULT = IN1K


@register_weights("pvt_v2_b5_cls")
class PVTv2B5Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.pvt_v2_b5_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``pvt_v2_b5.in1k`` (Wang et al., 2022; ~82.0M params, 83.8% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/pvt-v2-b5/resolve/main/IN1K/model.safetensors",
        sha256="d9636f599bff05ee39f8c74810ffd2bda99c75fef941278df32f9ba515bc4a09",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/pvt_v2_b5.in1k",
            "license": "apache-2.0",
            "num_params": 81_956_008,
            "metrics": {"ImageNet-1k": {"acc@1": 83.8}},
        },
    )
    DEFAULT = IN1K
