"""Pretrained-weight declarations for the GoogLeNet family.

The single paper-cited architecture (Szegedy et al., CVPR 2015) ships
one ImageNet-1k checkpoint converted from torchvision's
``GoogLeNet_Weights.IMAGENET1K_V1``: :class:`GoogLeNetWeights`.

The checkpoint uses the canonical 224 crop / 256 resize / bilinear
ImageNet eval pipeline.
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = ImageClassification(crop_size=224, resize_size=256, interpolation="bilinear")


@register_weights("googlenet_cls")
class GoogLeNetWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.googlenet_cls`.

    Single ImageNet-1k checkpoint converted from torchvision's
    ``GoogLeNet_Weights.IMAGENET1K_V1`` (Szegedy et al., 2015; ~13.0M
    params with auxiliary heads, 69.778% top-1).
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/googlenet/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="575008a21815fee0f9bdfd290add2bc1e63c6b8ec99550ccf2918bfa6a6064d7",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/GoogLeNet_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 13_004_888,
            "metrics": {"ImageNet-1k": {"acc@1": 69.778}},
        },
    )
    DEFAULT = IMAGENET1K_V1
