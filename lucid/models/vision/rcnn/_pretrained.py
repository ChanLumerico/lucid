"""Registry factories for R-CNN variants."""

from lucid.models._registry import register_model
from lucid.models.vision.rcnn._config import RCNNConfig
from lucid.models.vision.rcnn._model import RCNNForObjectDetection

# ---------------------------------------------------------------------------
# Base configurations (matching original paper)
# ---------------------------------------------------------------------------

# AlexNet backbone — original paper (CVPR 2014)
_CFG_ALEXNET = RCNNConfig(
    num_classes=80,
    in_channels=3,
    roi_size=227,
    dropout=0.5,
    score_thresh=0.05,
    nms_thresh=0.5,
    max_detections=300,
)

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _det(cfg: RCNNConfig, kw: dict[str, object]) -> RCNNForObjectDetection:
    return RCNNForObjectDetection(RCNNConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


# ---------------------------------------------------------------------------
# Object-detection factories
# ---------------------------------------------------------------------------


@register_model(
    task="object-detection",
    family="rcnn",
    model_type="rcnn",
    model_class=RCNNForObjectDetection,
    default_config=_CFG_ALEXNET,
)
def rcnn(
    pretrained: bool = False,
    **overrides: object,
) -> RCNNForObjectDetection:
    """R-CNN with AlexNet backbone (Girshick et al., CVPR 2014).

    Applies the AlexNet CNN to each region proposal crop independently and
    predicts class labels with bounding-box refinements.

    Region proposals must be supplied externally (e.g. selective search).
    """
    return _det(_CFG_ALEXNET, overrides)
