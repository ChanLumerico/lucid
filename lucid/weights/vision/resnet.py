"""Pretrained-weight enums for the ResNet family.

Each architecture exposes a :class:`~lucid.weights.WeightsEnum` listing
its tagged checkpoints hosted under the ``lucid-dl`` Hugging Face org.
These declarations are pure metadata — they import nothing from
:mod:`lucid.models`, keeping the dependency one-directional
(``models`` → ``weights``).
"""

from lucid.weights._base import WeightEntry, WeightsEnum
from lucid.weights._registry import register_weights
from lucid.weights._transforms import ImageClassification

_HUB = "https://huggingface.co/lucid-dl"


@register_weights("resnet_18")
class ResNet18Weights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.resnet_18`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{_HUB}/resnet-18/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="bb7eab3083c24be6364e32f1d37844a00c5e500fa48a83a91f750a7621d152cb",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=256),
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
