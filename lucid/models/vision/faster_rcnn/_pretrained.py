"""Registry factories for Faster R-CNN variants."""

from lucid.models._registry import register_model
from lucid.models.vision.faster_rcnn._config import FasterRCNNConfig
from lucid.models.vision.faster_rcnn._model import FasterRCNNForObjectDetection

_CFG_VGG16 = FasterRCNNConfig(
    num_classes=80,
    in_channels=3,
    rpn_anchor_sizes=(128, 256, 512),
    rpn_anchor_ratios=(0.5, 1.0, 2.0),
    rpn_pre_nms_top_n=2000,
    rpn_post_nms_top_n=1000,
    rpn_nms_thresh=0.7,
    rpn_min_size=16.0,
    rpn_score_thresh=0.0,
    rpn_fg_iou_thresh=0.7,
    rpn_bg_iou_thresh=0.3,
    roi_size=7,
    spatial_scale=1.0 / 16.0,
    roi_fg_iou_thresh=0.5,
    roi_bg_iou_thresh=0.5,
    dropout=0.5,
    bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
    score_thresh=0.05,
    nms_thresh=0.5,
    max_detections=300,
)


def _det(cfg: FasterRCNNConfig, kw: dict[str, object]) -> FasterRCNNForObjectDetection:
    return FasterRCNNForObjectDetection(
        FasterRCNNConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="object-detection",
    family="faster_rcnn",
    model_type="faster_rcnn",
    model_class=FasterRCNNForObjectDetection,
    default_config=_CFG_VGG16,
)
def faster_rcnn(
    pretrained: bool = False,
    **overrides: object,
) -> FasterRCNNForObjectDetection:
    """Faster R-CNN with VGG16 backbone (Ren et al., NeurIPS 2015).

    Introduces a Region Proposal Network (RPN) that shares the VGG16 backbone
    feature map with the detection head, replacing external selective-search
    proposals and enabling an end-to-end trainable pipeline.
    """
    return _det(_CFG_VGG16, overrides)
