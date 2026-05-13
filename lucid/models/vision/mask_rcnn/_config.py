"""Mask R-CNN configuration (He et al., 2017)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MaskRCNNConfig(ModelConfig):
    """Configuration for Mask R-CNN.

    Mask R-CNN (He et al., ICCV 2017) extends Faster R-CNN with:
      1. Feature Pyramid Network (FPN) backbone replacing single-scale VGG16.
      2. RoI Align replacing RoI Pool (eliminating quantisation misalignment).
      3. A parallel mask branch predicting K binary segmentation masks per RoI.

    Architecture overview:
      Image → ResNet-50 (C2–C5, strides 4/8/16/32)
        ↓ FPN (lateral convs + top-down merging) → [P2, P3, P4, P5, P6] (256ch)
      P2–P6 → RPN → proposals
      P2–P5 + proposals → FPN-level assignment → RoI Align (7×7) → 2-FC head
                                                → RoI Align (14×14) → Mask head

    Args:
        num_classes:  Foreground categories (background = class 0).
        in_channels:  Input image channels.

        -- Backbone/FPN --
        backbone_layers:   ResNet layer counts (default ResNet-50 = 3,4,6,3).
        fpn_out_channels:  Channel width of every FPN output level.

        -- RPN hyper-parameters --
        rpn_anchor_sizes:   One anchor size per FPN level (P2–P6).
        rpn_anchor_ratios:  Aspect ratios shared across all levels.
        rpn_pre_nms_top_n:  Proposals kept per level before NMS.
        rpn_post_nms_top_n: Proposals kept per image after NMS.
        rpn_nms_thresh:     RPN NMS IoU threshold.
        rpn_min_size:       Minimum proposal side length (feature pixels).
        rpn_score_thresh:   Minimum objectness score (post-sigmoid).
        rpn_fg_iou_thresh:  Anchor→GT IoU for foreground label.
        rpn_bg_iou_thresh:  Anchor→GT IoU upper bound for background.

        -- Detection head --
        roi_det_size:         RoI Align output for detection (7 → 7×7).
        roi_representation:   Hidden size of the 2-FC detection head.
        roi_fg_iou_thresh:    Proposal→GT IoU for fg assignment.
        roi_bg_iou_thresh:    Proposal→GT IoU upper bound for bg.
        bbox_reg_weights:     Per-component bbox delta scale.

        -- Mask head --
        roi_mask_size:       RoI Align output for the mask branch (14 → 14×14).
        mask_hidden_channels: Channel width inside the mask FCN.

        -- Inference --
        score_thresh:    Minimum final class score.
        nms_thresh:      Per-class NMS IoU threshold.
        max_detections:  Maximum detections returned per image.
        mask_thresh:     Binarisation threshold applied to mask sigmoid output.
    """

    model_type: ClassVar[str] = "mask_rcnn"

    num_classes: int = 80
    in_channels: int = 3

    # Backbone
    backbone_layers: tuple[int, int, int, int] = (3, 4, 6, 3)   # ResNet-50
    fpn_out_channels: int = 256

    # RPN
    rpn_anchor_sizes: tuple[int, ...] = (32, 64, 128, 256, 512)
    rpn_anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    rpn_pre_nms_top_n: int = 2000
    rpn_post_nms_top_n: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_min_size: float = 1.0
    rpn_score_thresh: float = 0.0
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3

    # Detection head
    roi_det_size: int = 7
    roi_representation: int = 1024
    roi_fg_iou_thresh: float = 0.5
    roi_bg_iou_thresh: float = 0.5
    bbox_reg_weights: tuple[float, float, float, float] = (10.0, 10.0, 5.0, 5.0)

    # Mask head
    roi_mask_size: int = 14
    mask_hidden_channels: int = 256

    # Inference
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 100
    mask_thresh: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_layers",  tuple(self.backbone_layers))
        object.__setattr__(self, "rpn_anchor_sizes",  tuple(self.rpn_anchor_sizes))
        object.__setattr__(self, "rpn_anchor_ratios", tuple(self.rpn_anchor_ratios))
        object.__setattr__(self, "bbox_reg_weights",  tuple(self.bbox_reg_weights))
