"""Pretrained-weight declarations for the MaxViT family.

Four paper-cited variants (Tu et al., ECCV 2022) ship ImageNet-1k
checkpoints converted from timm's ``maxvit_{tiny,small,base,large}_tf_224.in1k``
model zoo: :class:`MaxViTTinyWeights`, :class:`MaxViTSmallWeights`,
:class:`MaxViTBaseWeights`, :class:`MaxViTLargeWeights`.

The ``maxvit_xlarge`` variant is intentionally not shipped: timm only
publishes an ImageNet-22k (``in21k``) checkpoint for it, which has no
1000-class classification head and therefore cannot load into Lucid's
1k classifier.

Every checkpoint uses the timm MaxViT eval pipeline: 224 crop / 235
resize (``crop_pct = 0.95``) / bicubic interpolation / ImageNet stats.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=235, interpolation="bicubic")


@register_weights("maxvit_tiny_cls")
class MaxViTTinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.maxvit_tiny_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``maxvit_tiny_tf_224.in1k`` (Tu et al., 2022; ~30.9M params,
    83.62% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/maxvit-tiny/resolve/main/IN1K/model.safetensors",
        sha256="1c9df0b1280123b367ca8e6b6259647655fb6f43497b31c1d89a1b13c2df3394",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/maxvit_tiny_tf_224.in1k",
            "license": "apache-2.0",
            "num_params": 30_916_528,
            "metrics": {"ImageNet-1k": {"acc@1": 83.62}},
        },
    )
    DEFAULT = IN1K


@register_weights("maxvit_small_cls")
class MaxViTSmallWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.maxvit_small_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``maxvit_small_tf_224.in1k`` (Tu et al., 2022; ~68.9M params,
    84.45% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/maxvit-small/resolve/main/IN1K/model.safetensors",
        sha256="3d14c8ed4a14794d142beae56b141d40ee4e6ef2b015c51881ab2b53de6031be",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/maxvit_small_tf_224.in1k",
            "license": "apache-2.0",
            "num_params": 68_927_956,
            "metrics": {"ImageNet-1k": {"acc@1": 84.45}},
        },
    )
    DEFAULT = IN1K


@register_weights("maxvit_base_cls")
class MaxViTBaseWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.maxvit_base_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``maxvit_base_tf_224.in1k`` (Tu et al., 2022; ~119.5M params,
    84.95% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/maxvit-base/resolve/main/IN1K/model.safetensors",
        sha256="2378197e6a22c36becf4e85a0726f471603204778139575ddcb40926af05eac3",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/maxvit_base_tf_224.in1k",
            "license": "apache-2.0",
            "num_params": 119_467_708,
            "metrics": {"ImageNet-1k": {"acc@1": 84.95}},
        },
    )
    DEFAULT = IN1K


@register_weights("maxvit_large_cls")
class MaxViTLargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.maxvit_large_cls`.

    Single ImageNet-1k checkpoint converted from timm's
    ``maxvit_large_tf_224.in1k`` (Tu et al., 2022; ~211.8M params,
    85.17% top-1).
    """

    IN1K = WeightEntry(
        url=f"{HUB_BASE}/maxvit-large/resolve/main/IN1K/model.safetensors",
        sha256="f8570f0c4c0bfb4dc4e56038ed40846352cead50998f7d4a7f316c13e5242adf",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IN1K",
            "source": "timm/maxvit_large_tf_224.in1k",
            "license": "apache-2.0",
            "num_params": 211_785_560,
            "metrics": {"ImageNet-1k": {"acc@1": 85.17}},
        },
    )
    DEFAULT = IN1K
