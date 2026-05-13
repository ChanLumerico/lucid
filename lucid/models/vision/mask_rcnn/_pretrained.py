"""Registry factories for Mask R-CNN variants."""

from lucid.models._registry import register_model
from lucid.models.vision.mask_rcnn._config import MaskRCNNConfig
from lucid.models.vision.mask_rcnn._model import MaskRCNNForObjectDetection

_CFG_R50_FPN = MaskRCNNConfig(
    num_classes=80,
    in_channels=3,
    backbone_layers=(3, 4, 6, 3),
    fpn_out_channels=256,
    rpn_anchor_sizes=(32, 64, 128, 256, 512),
    rpn_anchor_ratios=(0.5, 1.0, 2.0),
    rpn_pre_nms_top_n=2000,
    rpn_post_nms_top_n=1000,
    rpn_nms_thresh=0.7,
    rpn_min_size=1.0,
    rpn_score_thresh=0.0,
    rpn_fg_iou_thresh=0.7,
    rpn_bg_iou_thresh=0.3,
    roi_det_size=7,
    roi_representation=1024,
    roi_fg_iou_thresh=0.5,
    roi_bg_iou_thresh=0.5,
    bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
    roi_mask_size=14,
    mask_hidden_channels=256,
    score_thresh=0.05,
    nms_thresh=0.5,
    max_detections=100,
    mask_thresh=0.5,
)


def _seg(cfg: MaskRCNNConfig, kw: dict[str, object]) -> MaskRCNNForObjectDetection:
    return MaskRCNNForObjectDetection(
        MaskRCNNConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="object-detection",
    family="mask_rcnn",
    model_type="mask_rcnn",
    model_class=MaskRCNNForObjectDetection,
    default_config=_CFG_R50_FPN,
)
def mask_rcnn(
    pretrained: bool = False,
    **overrides: object,
) -> MaskRCNNForObjectDetection:
    """Mask R-CNN with ResNet-50-FPN backbone (He et al., ICCV 2017).

    Extends Faster R-CNN with FPN, RoI Align, and a parallel mask head
    that predicts K binary segmentation masks per detected instance.
    """
    return _seg(_CFG_R50_FPN, overrides)
