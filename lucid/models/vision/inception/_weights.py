"""Pretrained-weight declarations for the Inception v3 family.

A single paper-cited architecture (Szegedy et al., CVPR 2016) converted
from torchvision's ``Inception_V3_Weights.IMAGENET1K_V1`` tag.  Unlike
the standard 224/256 ImageNet preset, Inception v3 evaluates at a
299-pixel centre crop resized from 342 (still ImageNet mean/std,
bilinear), so the transform is spelled out explicitly here to match the
source preset exactly.

The auxiliary classifier is dropped during conversion — the shipped
checkpoint targets :func:`lucid.models.inception_v3_cls`'s default
``aux_logits=False`` head (≈23.8 M parameters).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Inception v3 evaluation preset: 299 centre crop resized from 342, with
# the standard ImageNet mean/std and bilinear interpolation (sourced from
# torchvision's ``Inception_V3_Weights.IMAGENET1K_V1.transforms()``).
_PRESET = ImageClassification(crop_size=299, resize_size=342)


@register_weights("inception_v3_cls")
class InceptionV3Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.inception_v3_cls`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/inception-v3/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="5ea6e1c4e3758729ea55199e9eb50f38bef8ba39380d98dc865be7cfac91096a",
        num_classes=1000,
        transforms=_PRESET,
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/Inception_V3_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 23_834_568,
            "metrics": {"ImageNet-1k": {"acc@1": 77.294, "acc@5": 93.450}},
        },
    )
    DEFAULT = IMAGENET1K_V1
