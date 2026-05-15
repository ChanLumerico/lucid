"""Registry factories for DETR variants."""

from lucid.models._registry import register_model
from lucid.models.vision.detr._config import DETRConfig
from lucid.models.vision.detr._model import DETRForObjectDetection

_CFG_R50 = DETRConfig(
    num_classes=80,
    in_channels=3,
    backbone_layers=(3, 4, 6, 3),
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    num_bbox_layers=3,
    bbox_hidden_dim=256,
    score_thresh=0.7,
)

_CFG_R101 = DETRConfig(
    num_classes=80,
    in_channels=3,
    backbone_layers=(3, 4, 23, 3),  # ResNet-101
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    num_bbox_layers=3,
    bbox_hidden_dim=256,
    score_thresh=0.7,
)


def _det(cfg: DETRConfig, kw: dict[str, object]) -> DETRForObjectDetection:
    return DETRForObjectDetection(DETRConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


@register_model(
    task="object-detection",
    family="detr",
    model_type="detr",
    model_class=DETRForObjectDetection,
    default_config=_CFG_R50,
)
def detr_resnet50(
    pretrained: bool = False,
    **overrides: object,
) -> DETRForObjectDetection:
    """DETR with ResNet-50 backbone (Carion et al., ECCV 2020).

    Detection Transformer with 100 object queries, 6-layer encoder/decoder,
    d_model=256, 8 attention heads.  Trained with the set-prediction (Hungarian)
    loss — no anchors, no NMS.
    """
    return _det(_CFG_R50, overrides)


@register_model(
    task="object-detection",
    family="detr",
    model_type="detr",
    model_class=DETRForObjectDetection,
    default_config=_CFG_R101,
)
def detr_resnet101(
    pretrained: bool = False,
    **overrides: object,
) -> DETRForObjectDetection:
    """DETR with ResNet-101 backbone (Carion et al., ECCV 2020).

    Same transformer head configuration as the ResNet-50 variant, with a
    deeper backbone (23 layer-3 blocks instead of 6) for higher capacity.
    """
    return _det(_CFG_R101, overrides)
