"""Fast R-CNN configuration (Girshick, 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class FastRCNNConfig(ModelConfig):
    """Configuration for Fast R-CNN.

    Fast R-CNN (Girshick, ICCV 2015) runs the CNN on the **full image once**,
    then extracts per-proposal features with RoI Pooling on the shared feature
    map.  This is the key advance over R-CNN (one forward pass vs one per
    proposal).

    Default backbone: VGG16 conv layers (conv1_1 … conv5_3, pool5 removed).
    The feature map stride is 16 (four max-pool layers before pool5).

    Args:
        num_classes:     Foreground object classes.  Background is class 0,
                         giving (num_classes + 1) output logits.
        in_channels:     Input image channels (3 for RGB).
        roi_size:        RoI Pool output spatial size (paper: 7 → 7×7).
        spatial_scale:   Ratio of feature-map size to input image size.
                         For VGG16 with pool5 removed: 1/16.
        dropout:         Dropout probability after fc6 and fc7.
        bbox_reg_weights: Per-component weight scaling for bbox regression
                          targets (tx, ty, tw, th).  Matches the paper's
                          normalisation by (0, 0, 0, 0) mean / (0.1, 0.1,
                          0.2, 0.2) std, encoded here as multiplicative
                          weights.
        score_thresh:    Minimum class score at inference time.
        nms_thresh:      Per-class NMS IoU threshold.
        max_detections:  Maximum detections returned per image.
    """

    model_type: ClassVar[str] = "fast_rcnn"

    num_classes: int = 80
    in_channels: int = 3
    roi_size: int = 7
    spatial_scale: float = 1.0 / 16.0
    dropout: float = 0.5
    bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 300

    def __post_init__(self) -> None:
        object.__setattr__(self, "bbox_reg_weights", tuple(self.bbox_reg_weights))
