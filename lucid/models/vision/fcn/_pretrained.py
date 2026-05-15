"""Registry factories for FCN variants."""

from lucid.models._registry import register_model
from lucid.models.vision.fcn._config import FCNConfig
from lucid.models.vision.fcn._model import FCNForSemanticSegmentation

_CFG_RESNET50 = FCNConfig(
    num_classes=21,
    in_channels=3,
    backbone="resnet50",
    variant="fcn32s",
    classifier_hidden_channels=512,
    aux_hidden_channels=256,
    dropout=0.1,
)

_CFG_RESNET101 = FCNConfig(
    num_classes=21,
    in_channels=3,
    backbone="resnet101",
    variant="fcn32s",
    classifier_hidden_channels=512,
    aux_hidden_channels=256,
    dropout=0.1,
)


def _build(cfg: FCNConfig, kw: dict[str, object]) -> FCNForSemanticSegmentation:
    return FCNForSemanticSegmentation(
        FCNConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="semantic-segmentation",
    family="fcn",
    model_type="fcn",
    model_class=FCNForSemanticSegmentation,
    default_config=_CFG_RESNET50,
)
def fcn_resnet50(
    pretrained: bool = False,
    **overrides: object,
) -> FCNForSemanticSegmentation:
    """FCN with ResNet-50 backbone (Long et al., CVPR 2015).

    Standard configuration: ResNet-50 backbone with dilated convolutions
    (layer3 dilation=2, layer4 dilation=4), 21 output classes (Pascal VOC),
    512-channel FCN head, 256-channel auxiliary head.
    """
    return _build(_CFG_RESNET50, overrides)


@register_model(
    task="semantic-segmentation",
    family="fcn",
    model_type="fcn",
    model_class=FCNForSemanticSegmentation,
    default_config=_CFG_RESNET101,
)
def fcn_resnet101(
    pretrained: bool = False,
    **overrides: object,
) -> FCNForSemanticSegmentation:
    """FCN with ResNet-101 backbone (Long et al., CVPR 2015).

    Deeper backbone variant: ResNet-101 with 23 blocks in layer3 vs. 6.
    Same head configuration as fcn_resnet50.  Typically yields higher
    mean IoU at the cost of additional compute.
    """
    return _build(_CFG_RESNET101, overrides)
