"""Pretrained-weight declarations for the ResNet family.

Per-model weight enums live *with the model* (not in
:mod:`lucid.weights`).  :mod:`lucid.weights` is purely the porting
infrastructure — base classes, transforms, hub download, loading, and
the discovery registry; it knows nothing about specific architectures.
Each model family declares its checkpoints here by importing those
primitives.  Importing this module (which happens when
:mod:`lucid.models.vision.resnet` loads) registers the enums with the
discovery registry (:func:`lucid.weights.list_pretrained` /
:func:`lucid.weights.get_weight`).

These declarations are pure metadata — they pull only from
:mod:`lucid.weights`, never the reverse, so the dependency stays
one-directional (``models`` → ``weights``).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import WeightEntry, WeightsEnum, register_weights

_HUB = "https://huggingface.co/lucid-dl"


@register_weights("resnet_18")
class ResNet18Weights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.resnet_18_cls`.

    Currently ships a single ImageNet-1k checkpoint
    (:attr:`IMAGENET1K_V1`) — the canonical V1 weights distributed
    by the reference-framework model zoo, re-hosted under
    ``huggingface.co/lucid-dl/resnet-18`` with the official
    ``acc@1=69.758 / acc@5=89.078`` validation metrics.
    """

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
