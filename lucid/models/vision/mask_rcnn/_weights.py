"""Pretrained-weight declarations for the Mask R-CNN family.

The single paper-cited reference variant (He et al., ICCV 2017; Faster
R-CNN from Ren et al., NeurIPS 2015; FPN from Lin et al., CVPR 2017)
ships a COCO instance-segmentation checkpoint converted from the
reference ``MaskRCNN_ResNet50_FPN_Weights.COCO_V1``:
:class:`MaskRCNNResNet50FPNWeights`.

The checkpoint is the COCO detector at ``num_classes = 91`` (90
categories + background slot 0) and uses the reference detection eval
pipeline: longest-side 1333 resize + square pad / bilinear interpolation /
ImageNet normalisation (the :class:`~lucid.utils.transforms.Detection`
preset).
"""

from lucid.utils.transforms import Detection
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = Detection(max_size=1333)


@register_weights("mask_rcnn")
@register_weights("mask_rcnn_resnet50_fpn")
class MaskRCNNResNet50FPNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.mask_rcnn_resnet50_fpn`.

    Single COCO checkpoint converted from the reference
    ``MaskRCNN_ResNet50_FPN_Weights.COCO_V1`` (91 classes, box AP 37.9 /
    mask AP 34.6).
    """

    COCO_V1 = WeightEntry(
        url=(
            f"{HUB_BASE}/mask-rcnn-resnet-50-fpn/resolve/main/"
            "COCO_V1/model.safetensors"
        ),
        sha256="120a1648dda49f1799d8cc581bef8ec83ba679313602a80d5278409ab4e0ddcd",
        num_classes=91,
        transforms=_PRESET,
        meta={
            "tag": "COCO_V1",
            "source": "torchvision/MaskRCNN_ResNet50_FPN_Weights.COCO_V1",
            "license": "bsd-3-clause",
            "num_params": 44_401_393,
            "metrics": {"COCO": {"box mAP": 37.9, "mask mAP": 34.6}},
        },
    )
    DEFAULT = COCO_V1
