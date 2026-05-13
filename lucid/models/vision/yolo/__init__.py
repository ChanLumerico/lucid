"""YOLO family — You Only Look Once object detectors.

References
----------
- YOLOv1: Redmon et al., CVPR 2016
- YOLOv2: Redmon & Farhadi, CVPR 2017 (YOLO9000)
- YOLOv3: Redmon & Farhadi, arXiv 2018
- YOLOv4: Bochkovskiy et al., arXiv 2020
"""

from lucid.models.vision.yolo._v1 import (
    YOLOV1Config,
    YOLOV1ForObjectDetection,
    yolo_v1,
    yolo_v1_tiny,
)
from lucid.models.vision.yolo._v2 import (
    YOLOV2Config,
    YOLOV2ForObjectDetection,
    yolo_v2,
    yolo_v2_tiny,
)
from lucid.models.vision.yolo._v3 import (
    YOLOV3Config,
    YOLOV3ForObjectDetection,
    yolo_v3,
    yolo_v3_tiny,
)
from lucid.models.vision.yolo._v4 import (
    YOLOV4Config,
    YOLOV4ForObjectDetection,
    yolo_v4,
)

__all__ = [
    # YOLOv1
    "YOLOV1Config",
    "YOLOV1ForObjectDetection",
    "yolo_v1",
    "yolo_v1_tiny",
    # YOLOv2
    "YOLOV2Config",
    "YOLOV2ForObjectDetection",
    "yolo_v2",
    "yolo_v2_tiny",
    # YOLOv3
    "YOLOV3Config",
    "YOLOV3ForObjectDetection",
    "yolo_v3",
    "yolo_v3_tiny",
    # YOLOv4
    "YOLOV4Config",
    "YOLOV4ForObjectDetection",
    "yolo_v4",
]
