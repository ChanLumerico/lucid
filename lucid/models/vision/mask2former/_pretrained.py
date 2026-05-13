"""Registry factories for Mask2Former variants."""

from lucid.models._registry import register_model
from lucid.models.vision.mask2former._config import Mask2FormerConfig
from lucid.models.vision.mask2former._model import Mask2FormerForSemanticSegmentation

_CFG_R50 = Mask2FormerConfig(
    num_classes=150,
    in_channels=3,
    backbone_layers=(3, 4, 6, 3),
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    fpn_out_channels=256,
    num_feature_levels=3,
)

_CFG_R101 = Mask2FormerConfig(
    num_classes=150,
    in_channels=3,
    backbone_layers=(3, 4, 23, 3),  # ResNet-101
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    fpn_out_channels=256,
    num_feature_levels=3,
)


def _build(
    cfg: Mask2FormerConfig, kw: dict[str, object]
) -> Mask2FormerForSemanticSegmentation:
    return Mask2FormerForSemanticSegmentation(
        Mask2FormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_R50,
)
def mask2former_resnet50(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with ResNet-50 backbone (Cheng et al., CVPR 2022).

    Masked-attention transformer decoder with multi-scale FPN cross-attention
    (3 levels: P3/P4/P5).  100 queries, d_model=256, 6-layer decoder,
    8 attention heads.  ADE20K default (150 classes).
    """
    return _build(_CFG_R50, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_R101,
)
def mask2former_resnet101(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with ResNet-101 backbone (Cheng et al., CVPR 2022).

    Deeper backbone (23 layer3 blocks) with same transformer head as
    the ResNet-50 variant.  Improved accuracy at higher compute cost.
    """
    return _build(_CFG_R101, overrides)
