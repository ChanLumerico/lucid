"""Pretrained-weight declarations for the Faster R-CNN family.

The single paper-cited reference variant (Ren et al., NeurIPS 2015; FPN
from Lin et al., CVPR 2017) ships a COCO detection checkpoint converted
from the reference ``FasterRCNN_ResNet50_FPN_Weights.COCO_V1``:
:class:`FasterRCNNResNet50FPNWeights`.

The checkpoint is the COCO detector at ``num_classes = 91`` (90
categories + background slot 0) and uses the reference detection eval
pipeline: longest-side 1333 resize + square pad / bilinear interpolation /
ImageNet normalisation (the :class:`~lucid.utils.transforms.Detection`
preset).
"""

from lucid.utils.transforms import Detection
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

_PRESET = Detection(max_size=1333)


@register_weights("faster_rcnn")
@register_weights("faster_rcnn_resnet50_fpn")
class FasterRCNNResNet50FPNWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.faster_rcnn_resnet50_fpn`.

    Single COCO checkpoint converted from the reference
    ``FasterRCNN_ResNet50_FPN_Weights.COCO_V1`` (91 classes, box AP 37.0).
    """

    COCO_V1 = WeightEntry(
        url=(
            f"{HUB_BASE}/faster-rcnn-resnet-50-fpn/resolve/main/"
            "COCO_V1/model.safetensors"
        ),
        sha256="cc8d2c79125d73432bee0eadd049f2871368a90a7bb0b0c5cba63af3c29f3205",
        num_classes=91,
        transforms=_PRESET,
        meta={
            "tag": "COCO_V1",
            "source": "torchvision/FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
            "license": "bsd-3-clause",
            "num_params": 41_755_286,
            "metrics": {"COCO": {"box mAP": 37.0}},
        },
    )
    DEFAULT = COCO_V1
