"""Mask R-CNN — instance segmentation (He et al., ICCV 2017).

Paper: "Mask R-CNN"
"""

from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig
from lucid.models.vision.mask_rcnn._model import MaskRCNNForObjectDetection
from lucid.models.vision.mask_rcnn._pretrained import mask_rcnn

__all__ = [
    "MaskRCNNConfig",
    "MaskRCNNForObjectDetection",
    "mask_rcnn",
]
