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
    r"""Mask R-CNN with ResNet-50-FPN backbone (He et al., ICCV 2017).

    Builds a :class:`MaskRCNNForObjectDetection` with the paper-cited
    ResNet-50 + FPN topology: 256-channel FPN over ``C2-C5``, 5-scale
    anchors (sizes 32 / 64 / 128 / 256 / 512 over P2-P6 with ratios
    0.5 / 1.0 / 2.0), 7x7 RoI Align detection head, and 14x14 RoI Align
    -> four-conv -> deconv 28x28 mask head.  Reaches COCO test-dev mask
    AP of 33.6% (paper Table 1, R50-FPN row) and is the canonical
    baseline for instance / panoptic segmentation research.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`MaskRCNNConfig` —
        common knobs include ``num_classes``, ``backbone_layers``
        (e.g. ``(3, 4, 23, 3)`` for ResNet-101), ``fpn_out_channels``,
        ``max_detections``, and ``mask_thresh``.

    Returns
    -------
    MaskRCNNForObjectDetection
        Detector with the Mask R-CNN ResNet-50-FPN configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See He et al., "Mask R-CNN", ICCV 2017 (arXiv:1703.06870).  The two
    key contributions over Faster R-CNN are RoI Align (replacing RoI Pool
    to remove quantisation, which costs ~3 AP on small objects) and the
    parallel mask head, which decouples mask and class prediction to
    avoid competition between the two tasks.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask_rcnn import mask_rcnn
    >>> model = mask_rcnn(num_classes=80)
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.pred_masks.shape[-2:]
    (28, 28)
    """
    return _seg(_CFG_R50_FPN, overrides)
