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

# NOTE(weights-upload): ``sha256`` is filled in once the converted
# checkpoint is uploaded to the Hub (see ``tools/convert_weights``).
# Until then the enum structure + resolution work, but a live
# ``pretrained=True`` download will fail SHA verification by design.
_PENDING_SHA = ""


@register_weights("resnet_18")
class ResNet18Weights(WeightsEnum):
    """Pretrained weights for :func:`lucid.models.resnet_18`."""

    IMAGENET1K_V1 = WeightEntry(
        url=f"{_HUB}/resnet-18/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256=_PENDING_SHA,
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=256),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/ResNet18_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "recipe": (
                "https://github.com/pytorch/vision/tree/main/"
                "references/classification#resnet"
            ),
            "num_params": 11_689_512,
            "gflops": 1.814,
            "file_size_mb": 44.7,
            "metrics": {"ImageNet-1k": {"acc@1": 69.758, "acc@5": 89.078}},
        },
    )
    DEFAULT = IMAGENET1K_V1
