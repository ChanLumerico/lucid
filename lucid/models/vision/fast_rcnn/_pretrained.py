"""Registry factories for Fast R-CNN variants."""

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


def _det(
    cfg: FastRCNNConfig, kw: dict[str, object]
) -> FastRCNNForObjectDetection:
    return FastRCNNForObjectDetection(
        FastRCNNConfig(**{**cfg.__dict__, **kw}) if kw else cfg
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
    """Fast R-CNN with VGG16 backbone (Girshick, ICCV 2015).

    Applies the backbone CNN once to the full image, then extracts per-proposal
    features via RoI Pool (7×7) on the shared feature map.
    """
    return _det(_CFG_VGG16, overrides)
