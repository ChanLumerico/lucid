"""Faster R-CNN — object detection (Ren et al., NeurIPS 2015).

Paper: "Faster R-CNN: Towards Real-Time Object Detection with
        Region Proposal Networks"
"""

from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig
from lucid.models.vision.faster_rcnn._model import FasterRCNNForObjectDetection
from lucid.models.vision.faster_rcnn._pretrained import faster_rcnn

__all__ = [
    "FasterRCNNConfig",
    "FasterRCNNForObjectDetection",
    "faster_rcnn",
]
