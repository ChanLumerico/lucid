"""Pretrained-weight declarations for the YOLO family.

Four COCO detectors converted straight from the original darknet
``.weights`` releases by :mod:`tools.convert_weights.yolo` — no
intermediate framework is involved, so these are the authors' own trained
parameters rather than a reimplementation's.  Each checkpoint's tensor
budget matches its darknet blob exactly (see the converter's module
docstring for the arithmetic), and all four reproduce darknet's published
detections on ``data/dog.jpg``.

Preprocessing is darknet's, which is unusual for this zoo: letterbox to a
square canvas and feed raw :math:`[0, 1]` pixels.  No dataset mean/std is
subtracted anywhere in darknet's pipeline, so the presets below pass
``mean = 0`` / ``std = 1``.  The letterbox pad value is 0.5 grey.

``yolo_v1`` has no entry here: the weights pjreddie published under
``yolov1/`` are for a different configuration than the paper's detector,
and they do not fit this family's ``yolo_v1`` topology.
"""

from lucid.utils.transforms import Detection
from lucid.weights import HUB_BASE, WeightEntry, WeightsEnum, register_weights

# Darknet letterbox + [0, 1] pixels.  v2 / v4 ship at 608, the v3 pair at 416
# — the resolution each release quotes its COCO mAP at.
_PRESET_416 = Detection(max_size=416, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
_PRESET_608 = Detection(max_size=608, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))


@register_weights("yolo_v2")
class YOLOV2Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v2`.

    Single COCO checkpoint (:attr:`COCO_2014`) converted from darknet's
    ``yolov2.weights``, hosted under ``huggingface.co/lucid-dl/yolo-v2``.
    Evaluated at :math:`608 \times 608`, where the darknet release reports
    ``mAP@0.5 = 48.1``.
    """

    COCO_2014 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v2/resolve/main/COCO_2014/model.safetensors",
        sha256="369e05ba62b16cff6a51026e8ddd73d0717712d9550ce35712551bce1a377762",
        num_classes=80,
        transforms=_PRESET_608,
        meta={
            "tag": "COCO_2014",
            "source": "darknet/yolov2.weights",
            "license": "other",
            "num_params": 50_962_889,
            "file_size_mb": 194.5,
            "metrics": {"COCO": {"mAP@0.5": 48.1}},
        },
    )
    DEFAULT = COCO_2014


@register_weights("yolo_v3")
class YOLOV3Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v3`.

    Single COCO checkpoint (:attr:`COCO_2014`) converted from darknet's
    ``yolov3.weights``, hosted under ``huggingface.co/lucid-dl/yolo-v3``.
    Evaluated at :math:`416 \times 416`, where the darknet release reports
    ``mAP@0.5 = 55.3``.
    """

    COCO_2014 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v3/resolve/main/COCO_2014/model.safetensors",
        sha256="61e5d156ee8445345f7588c36d38cc006e797024df3f33b2c06c156d8fc44f81",
        num_classes=80,
        transforms=_PRESET_416,
        meta={
            "tag": "COCO_2014",
            "source": "darknet/yolov3.weights",
            "license": "other",
            "num_params": 61_949_149,
            "file_size_mb": 236.57,
            "metrics": {"COCO": {"mAP@0.5": 55.3}},
        },
    )
    DEFAULT = COCO_2014


@register_weights("yolo_v3_tiny")
class YOLOV3TinyWeights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v3_tiny`.

    Single COCO checkpoint (:attr:`COCO_2014`) converted from darknet's
    ``yolov3-tiny.weights``, hosted under
    ``huggingface.co/lucid-dl/yolo-v3-tiny``.  Evaluated at
    :math:`416 \times 416`, where the darknet release reports
    ``mAP@0.5 = 33.1``.
    """

    COCO_2014 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v3-tiny/resolve/main/COCO_2014/model.safetensors",
        sha256="35e8f39a87a0302df9999c2ceaa77b9ae9dd69311cea4611c182056f8b6a63ad",
        num_classes=80,
        transforms=_PRESET_416,
        meta={
            "tag": "COCO_2014",
            "source": "darknet/yolov3-tiny.weights",
            "license": "other",
            "num_params": 8_852_366,
            "file_size_mb": 33.8,
            "metrics": {"COCO": {"mAP@0.5": 33.1}},
        },
    )
    DEFAULT = COCO_2014


@register_weights("yolo_v4")
class YOLOV4Weights(WeightsEnum):
    r"""Pretrained weights for :func:`lucid.models.yolo_v4`.

    Single COCO checkpoint (:attr:`COCO_2017`) converted from the AlexeyAB
    darknet release's ``yolov4.weights``, hosted under
    ``huggingface.co/lucid-dl/yolo-v4``.  Evaluated at
    :math:`608 \times 608`, where that release reports
    ``mAP@0.5 = 65.7`` on COCO ``test-dev``.
    """

    COCO_2017 = WeightEntry(
        url=f"{HUB_BASE}/yolo-v4/resolve/main/COCO_2017/model.safetensors",
        sha256="f624703802e4c933c3f8d5ce4de4e8686f2a71466a7ad27321484c4b31dd0530",
        num_classes=80,
        transforms=_PRESET_608,
        meta={
            "tag": "COCO_2017",
            "source": "darknet/yolov4.weights",
            "license": "other",
            "num_params": 64_363_101,
            "file_size_mb": 245.85,
            "metrics": {"COCO": {"mAP@0.5": 65.7}},
        },
    )
    DEFAULT = COCO_2017
