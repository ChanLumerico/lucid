"""Pretrained-weight declarations for the AlexNet family.

Per-model weight enums live *with the model* (not in
:mod:`lucid.weights`).  :mod:`lucid.weights` is purely the porting
infrastructure — base classes, transforms, hub download, loading, and
the discovery registry; it knows nothing about specific architectures.
Each model family declares its checkpoints here by importing those
primitives.  Importing this module (which happens when
:mod:`lucid.models.vision.alexnet` loads) registers the enums with the
discovery registry (:func:`lucid.weights.list_pretrained` /
:func:`lucid.weights.get_weight`).

These declarations are pure metadata — they pull only from
:mod:`lucid.weights`, never the reverse, so the dependency stays
one-directional (``models`` → ``weights``).
"""

from lucid.utils.transforms import ImageClassification
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights


@register_weights("alexnet_cls")
class AlexNetWeights(WeightsEnum):
    r"""Pretrained weight tags for :func:`lucid.models.alexnet_cls`.

    Currently ships a single ImageNet-1k checkpoint
    (:attr:`IMAGENET1K_V1`) — the canonical single-stream OWT weights
    distributed by the reference-framework model zoo, re-hosted under
    ``huggingface.co/lucid-dl/alexnet`` with the published
    ``acc@1=56.522 / acc@5=79.066`` validation metrics.
    """

    IMAGENET1K_V1 = WeightEntry(
        url=f"{HUB_BASE}/alexnet/resolve/main/IMAGENET1K_V1/model.safetensors",
        sha256="6c5894a4d55819004bc0e787fc6195e14bb56d21276924fde3f2cd4f7aeab6a2",
        num_classes=1000,
        transforms=ImageClassification(crop_size=224, resize_size=256),
        meta={
            "tag": "IMAGENET1K_V1",
            "source": "torchvision/AlexNet_Weights.IMAGENET1K_V1",
            "license": "bsd-3-clause",
            "num_params": 61_100_840,
            "gflops": 0.714,
            "file_size_mb": 233.1,
            "metrics": {"ImageNet-1k": {"acc@1": 56.522, "acc@5": 79.066}},
        },
    )
    DEFAULT = IMAGENET1K_V1
