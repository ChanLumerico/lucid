"""Registry factories for Fast R-CNN variants."""

from dataclasses import replace
from typing import Any, cast

from lucid.models._registry import register_model
from lucid.models.vision.fast_rcnn._config import FastRCNNConfig
from lucid.models.vision.fast_rcnn._model import FastRCNNForObjectDetection

_CFG_VGG16 = FastRCNNConfig(
    num_classes=80,
    in_channels=3,
    roi_size=7,
    spatial_scale=1.0 / 16.0,
    dropout=0.5,
    bbox_reg_weights=(10.0, 10.0, 5.0, 5.0),
    score_thresh=0.05,
    nms_thresh=0.5,
    max_detections=300,
)


def _det(cfg: FastRCNNConfig, kw: dict[str, object]) -> FastRCNNForObjectDetection:
    return FastRCNNForObjectDetection(
        replace(cfg, **cast(dict[str, Any], kw)) if kw else cfg
    )


@register_model(
    task="object-detection",
    family="fast_rcnn",
    model_type="fast_rcnn",
    model_class=FastRCNNForObjectDetection,
    default_config=_CFG_VGG16,
)
def fast_rcnn(
    pretrained: bool = False,
    **overrides: object,
) -> FastRCNNForObjectDetection:
    r"""Fast R-CNN with VGG16 backbone (Girshick, ICCV 2015).

    Builds a :class:`FastRCNNForObjectDetection` with the paper-cited
    VGG16 topology: shared backbone applied once to the full image, then
    7x7 RoI Pool over the stride-16 feature map followed by two 4096-dim
    FC layers and sibling K+1 / 4K heads.  Approximately 138M parameters
    (most in the FC layers) and ~146x faster training than R-CNN while
    reaching the same VOC2007 mAP of 66.9% (paper Table 1).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`FastRCNNConfig`
        (``num_classes``, ``score_thresh``, ``nms_thresh``,
        ``max_detections``, ``bbox_reg_weights``, ``dropout``, ...).

    Returns
    -------
    FastRCNNForObjectDetection
        Detector with the Fast R-CNN VGG16 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Girshick, "Fast R-CNN", ICCV 2015 (arXiv:1504.08083).  Region
    proposals must still be supplied externally — the Faster R-CNN
    successor folds proposal generation into the network via the Region
    Proposal Network (RPN).  The default ``spatial_scale = 1/16`` matches
    VGG16's four max-pool layers before pool5.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.fast_rcnn import fast_rcnn
    >>> model = fast_rcnn(num_classes=80)
    >>> x = lucid.randn(1, 3, 600, 800)
    >>> proposals = [lucid.tensor(
    ...     [[10.0, 10.0, 200.0, 200.0],
    ...      [50.0, 60.0, 300.0, 280.0]])]
    >>> out = model(x, proposals)
    >>> out.logits.shape, out.pred_boxes.shape
    ((2, 81), (2, 80, 4))
    """
    return _det(_CFG_VGG16, overrides)
