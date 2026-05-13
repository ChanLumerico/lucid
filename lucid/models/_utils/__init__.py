"""lucid.models._utils — task-specific model utilities.

Sub-modules
-----------
_common         Cross-task helpers (make_divisible …)
_classification Image-classification helpers (LayerScale …)
_detection      Detection helpers (box_ops, NMS, anchors, RoI ops, FPN, RPN …)

Most callers should import directly from the sub-module:

    from lucid.models._utils._common import make_divisible
    from lucid.models._utils._classification import LayerScale
    from lucid.models._utils._detection import (
        box_iou, nms, AnchorGenerator, FPN, RPN, RoIHead,
    )

Alternatively the most-used names are re-exported here for convenience:

    from lucid.models._utils import make_divisible, LayerScale
"""

from lucid.models._utils._common import make_divisible
from lucid.models._utils._classification import LayerScale
from lucid.models._utils._detection import (
    # Box operations
    box_area,
    box_iou,
    generalized_box_iou,
    box_xyxy_to_cxcywh,
    box_cxcywh_to_xyxy,
    clip_boxes_to_image,
    remove_small_boxes,
    encode_boxes,
    decode_boxes,
    # NMS
    nms,
    batched_nms,
    # Anchors
    AnchorGenerator,
    # RoI ops
    roi_align,
    roi_pool,
    # Shared modules
    FPN,
    RPN,
    RoIHead,
)

__all__ = [
    # common
    "make_divisible",
    # classification
    "LayerScale",
    # detection — box ops
    "box_area",
    "box_iou",
    "generalized_box_iou",
    "box_xyxy_to_cxcywh",
    "box_cxcywh_to_xyxy",
    "clip_boxes_to_image",
    "remove_small_boxes",
    "encode_boxes",
    "decode_boxes",
    # detection — NMS
    "nms",
    "batched_nms",
    # detection — anchors
    "AnchorGenerator",
    # detection — RoI ops
    "roi_align",
    "roi_pool",
    # detection — shared modules
    "FPN",
    "RPN",
    "RoIHead",
]
