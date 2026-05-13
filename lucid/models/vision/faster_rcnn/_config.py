"""Faster R-CNN configuration (Ren et al., 2015)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class FasterRCNNConfig(ModelConfig):
    """Configuration for Faster R-CNN.

    Faster R-CNN (Ren et al., NeurIPS 2015) replaces external proposal methods
    (selective search) with a learned Region Proposal Network (RPN) that shares
    convolutional features with the detection head.  This makes the entire
    pipeline end-to-end trainable and near real-time.

    Architecture overview:
      Image → VGG16 backbone (conv1_1 … conv5_3, stride 16) → feature map
      Feature map → RPN → region proposals
      Feature map + proposals → RoI Pool (7×7) → FC head → cls + bbox

    Args:
        num_classes:      Foreground object classes (background = class 0).
        in_channels:      Input image channels.

        -- RPN hyper-parameters --
        rpn_anchor_sizes:    Anchor base sizes per FPN level (single-level
                             for VGG backbone: one set of anchors).
        rpn_anchor_ratios:   Aspect ratios for anchor generation.
        rpn_pre_nms_top_n:   Proposals kept per level before NMS.
        rpn_post_nms_top_n:  Proposals kept per image after NMS.
        rpn_nms_thresh:      RPN NMS IoU threshold.
        rpn_min_size:        Minimum proposal side length (pixels).
        rpn_score_thresh:    Minimum RPN objectness score.
        rpn_fg_iou_thresh:   Anchor→GT IoU for foreground assignment.
        rpn_bg_iou_thresh:   Anchor→GT IoU upper bound for background.

        -- RoI head hyper-parameters --
        roi_size:            RoI Pool output spatial size (7 → 7×7).
        spatial_scale:       Feature-map / image scale ratio (1/16 for VGG16).
        roi_fg_iou_thresh:   Proposal→GT IoU for fg assignment in RoI head.
        roi_bg_iou_thresh:   Proposal→GT IoU upper bound for bg in RoI head.
        dropout:             Dropout after fc6 / fc7.
        bbox_reg_weights:    Per-component bbox delta scale (paper §3.1).

        -- Inference hyper-parameters --
        score_thresh:     Minimum final class score.
        nms_thresh:       Per-class NMS IoU threshold.
        max_detections:   Maximum detections returned per image.
    """

    model_type: ClassVar[str] = "faster_rcnn"

    num_classes: int = 80
    in_channels: int = 3

    # RPN
    rpn_anchor_sizes: tuple[int, ...] = (128, 256, 512)
    rpn_anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    rpn_pre_nms_top_n: int = 2000
    rpn_post_nms_top_n: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_min_size: float = 16.0
    rpn_score_thresh: float = 0.0
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3

    # RoI head
    roi_size: int = 7
    spatial_scale: float = 1.0 / 16.0
    roi_fg_iou_thresh: float = 0.5
    roi_bg_iou_thresh: float = 0.5
    dropout: float = 0.5
    bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)

    # Inference
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 300

    def __post_init__(self) -> None:
        object.__setattr__(self, "rpn_anchor_sizes",  tuple(self.rpn_anchor_sizes))
        object.__setattr__(self, "rpn_anchor_ratios", tuple(self.rpn_anchor_ratios))
        object.__setattr__(self, "bbox_reg_weights",  tuple(self.bbox_reg_weights))
