"""R-CNN configuration (Girshick et al., 2014)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class RCNNConfig(ModelConfig):
    """Configuration for R-CNN.

    R-CNN (Girshick et al., CVPR 2014) applies a CNN independently to each
    region proposal (warped to a fixed ``roi_size × roi_size`` crop), then
    feeds the flattened pool5 features through two FC layers before the
    classification and bounding-box regression heads.

    Original paper uses AlexNet (5 conv + 3 FC) as the backbone.  The
    effective output of pool5 for a 227 × 227 input is 6 × 6 × 256 = 9 216.

    Args:
        num_classes:     Number of foreground object classes.
                         Background is automatically added as class 0,
                         giving (num_classes + 1) output logits.
        in_channels:     Number of input image channels (3 for RGB).
        roi_size:        Each region proposal is warped to this square size
                         before being forwarded through the backbone.
                         Original paper: 227.
        dropout:         Dropout probability applied after fc6 and fc7.
        score_thresh:    Minimum class-score threshold applied during
                         post-processing (after softmax).  Boxes whose
                         max class score is below this value are discarded.
        nms_thresh:      IoU threshold for per-class NMS at inference time.
        max_detections:  Maximum number of detections returned per image
                         after NMS.
    """

    model_type: ClassVar[str] = "rcnn"

    num_classes: int = 80
    in_channels: int = 3
    roi_size: int = 227
    dropout: float = 0.5
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 300
