"""Registry factories for MaskFormer variants."""

from lucid.models._registry import register_model
from lucid.models.vision.maskformer._config import MaskFormerConfig
from lucid.models.vision.maskformer._model import MaskFormerForSemanticSegmentation

_CFG_R50 = MaskFormerConfig(
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
)

_CFG_R101 = MaskFormerConfig(
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
)


def _build(
    cfg: MaskFormerConfig, kw: dict[str, object]
) -> MaskFormerForSemanticSegmentation:
    return MaskFormerForSemanticSegmentation(
        MaskFormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="semantic-segmentation",
    family="maskformer",
    model_type="maskformer",
    model_class=MaskFormerForSemanticSegmentation,
    default_config=_CFG_R50,
)
def maskformer_resnet50(
    pretrained: bool = False,
    **overrides: object,
) -> MaskFormerForSemanticSegmentation:
    """MaskFormer with ResNet-50 backbone (Cheng et al., NeurIPS 2021).

    Mask-classification segmentation: 100 queries, 6-layer transformer decoder,
    d_model=256, 8 attention heads.  ADE20K default (150 classes).
    """
    return _build(_CFG_R50, overrides)


@register_model(
    task="semantic-segmentation",
    family="maskformer",
    model_type="maskformer",
    model_class=MaskFormerForSemanticSegmentation,
    default_config=_CFG_R101,
)
def maskformer_resnet101(
    pretrained: bool = False,
    **overrides: object,
) -> MaskFormerForSemanticSegmentation:
    """MaskFormer with ResNet-101 backbone (Cheng et al., NeurIPS 2021).

    Deeper backbone variant (23 layer3 blocks vs 6 for ResNet-50).
    Same transformer head configuration as the ResNet-50 variant.
    """
    return _build(_CFG_R101, overrides)
