"""Pretrained-weight declarations for the Mask2Former family.

Four paper-cited Swin-backbone variants (Cheng et al., CVPR 2022) ship
ADE20k semantic-segmentation checkpoints converted from the
``facebook/mask2former-swin-{tiny,small,base,large}-ade-semantic``
``Mask2FormerForUniversalSegmentation`` checkpoints:
:class:`Mask2FormerSwinTinyWeights` / ``...Small`` / ``...Base`` / ``...Large``.

Each checkpoint is the 150-class ADE20k segmenter (mask-classification
formulation; ``class_predictor`` is ``(151, 256)`` = 150 foreground + 1
no-object).  The eval pipeline uses the 384² ADE20k semantic crop with
ImageNet normalisation sourced from the upstream image processor (the
:class:`~lucid.utils.transforms.Segmentation` preset), and the model's
semantic output drops the no-object slot before the per-query
softmax ⊗ sigmoid summation.
"""

from lucid.utils.transforms import Segmentation
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = Segmentation(
    crop_size=384,
    resize_size=384,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


@register_weights("mask2former_swin_tiny")
class Mask2FormerSwinTinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mask2former_swin_tiny`.

    Single ADE20k semantic checkpoint converted from
    ``facebook/mask2former-swin-tiny-ade-semantic`` (Cheng et al., 2022;
    ~47.4M params, 47.7 mIoU on the ADE20k validation set).
    """

    ADE20K = WeightEntry(
        url=f"{HUB_BASE}/mask2former-swin-tiny-ade/resolve/main/"
        "ADE20K/model.safetensors",
        sha256="f374059bb4a26507b7c243c86684591724b07712616a665cd5beb3fcdf662b8f",
        num_classes=150,
        transforms=_PRESET,
        meta={
            "tag": "ADE20K",
            "source": "facebook/mask2former-swin-tiny-ade-semantic",
            "license": "other",
            "num_params": 47_441_169,
            "metrics": {"ADE20K": {"mIoU": 47.7}},
        },
    )
    DEFAULT = ADE20K


@register_weights("mask2former_swin_small")
class Mask2FormerSwinSmallWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mask2former_swin_small`.

    Single ADE20k semantic checkpoint converted from
    ``facebook/mask2former-swin-small-ade-semantic`` (Cheng et al., 2022;
    ~68.7M params, 51.3 mIoU on the ADE20k validation set).
    """

    ADE20K = WeightEntry(
        url=f"{HUB_BASE}/mask2former-swin-small-ade/resolve/main/"
        "ADE20K/model.safetensors",
        sha256="94bc512d971caaa956c541a54e26f11d1eea68fd36af23c3df3b94576705335e",
        num_classes=150,
        transforms=_PRESET,
        meta={
            "tag": "ADE20K",
            "source": "facebook/mask2former-swin-small-ade-semantic",
            "license": "other",
            "num_params": 68_815_312,
            "metrics": {"ADE20K": {"mIoU": 51.3}},
        },
    )
    DEFAULT = ADE20K


@register_weights("mask2former_swin_base")
class Mask2FormerSwinBaseWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mask2former_swin_base`.

    Single ADE20k semantic checkpoint converted from
    ``facebook/mask2former-swin-base-ade-semantic`` (Cheng et al., 2022;
    ~107M params, 53.9 mIoU on the ADE20k validation set).
    """

    ADE20K = WeightEntry(
        url=f"{HUB_BASE}/mask2former-swin-base-ade/resolve/main/"
        "ADE20K/model.safetensors",
        sha256="3185a471a980cdb9a53c22d7995a8bb6afe4363723bd1d9bfef8d9bf7ebee447",
        num_classes=150,
        transforms=_PRESET,
        meta={
            "tag": "ADE20K",
            "source": "facebook/mask2former-swin-base-ade-semantic",
            "license": "other",
            "num_params": 107_420_006,
            "metrics": {"ADE20K": {"mIoU": 53.9}},
        },
    )
    DEFAULT = ADE20K


@register_weights("mask2former_swin_large")
class Mask2FormerSwinLargeWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mask2former_swin_large`.

    Single ADE20k semantic checkpoint converted from
    ``facebook/mask2former-swin-large-ade-semantic`` (Cheng et al., 2022;
    ~215M params, 56.1 mIoU on the ADE20k validation set — the headline
    paper result).
    """

    ADE20K = WeightEntry(
        url=f"{HUB_BASE}/mask2former-swin-large-ade/resolve/main/"
        "ADE20K/model.safetensors",
        sha256="b6979833993857fbe2f20d1c8a207a465278c6974e300f49ff353b4da732d34c",
        num_classes=150,
        transforms=_PRESET,
        meta={
            "tag": "ADE20K",
            "source": "facebook/mask2former-swin-large-ade-semantic",
            "license": "other",
            "num_params": 215_986_594,
            "metrics": {"ADE20K": {"mIoU": 56.1}},
        },
    )
    DEFAULT = ADE20K
