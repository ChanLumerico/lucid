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
    r"""Faster R-CNN with VGG16 backbone (Ren et al., NeurIPS 2015).

    Builds a :class:`FasterRCNNForObjectDetection` with the paper-cited
    VGG16 + 3-scale/3-ratio anchor configuration: anchors at sizes
    ``(128, 256, 512)`` and aspect ratios ``(0.5, 1.0, 2.0)`` at every
    stride-16 cell (9 anchors / cell), 2000 / 1000 pre / post-NMS proposals
    at training time, and a 7x7 RoI Pool detection head matching Fast R-CNN.
    PASCAL VOC 2007 test mAP of 73.2% with selective-search replaced by
    the learned RPN (paper Table 6) and ~5 fps inference on a Titan X.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`FasterRCNNConfig` —
        common knobs include ``num_classes``, ``rpn_anchor_sizes`` /
        ``rpn_anchor_ratios`` for different scale priors,
        ``rpn_post_nms_top_n`` to vary the proposal budget, and
        ``score_thresh`` / ``nms_thresh`` for inference filtering.

    Returns
    -------
    FasterRCNNForObjectDetection
        Detector with the Faster R-CNN VGG16 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Ren et al., "Faster R-CNN: Towards Real-Time Object Detection
    with Region Proposal Networks", NeurIPS 2015 (arXiv:1506.01497).
    The key contribution is replacing the external selective-search
    proposal stage with a learned RPN that shares the backbone — at the
    cost of one extra 3x3 conv plus two 1x1 sibling convs per cell.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.faster_rcnn import faster_rcnn
    >>> model = faster_rcnn(num_classes=80)
    >>> x = lucid.randn(1, 3, 800, 800)
    >>> out = model(x)
    >>> out.logits.shape[-1]
    81
    """
    return _det(_CFG_VGG16, overrides)
