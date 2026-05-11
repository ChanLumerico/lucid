"""Registry factories for all SE-ResNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.senet._config import SENetConfig
from lucid.models.vision.senet._model import SENet, SENetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_18 = SENetConfig(block_type="basic", layers=(2, 2, 2, 2))
_CFG_34 = SENetConfig(block_type="basic", layers=(3, 4, 6, 3))
_CFG_50 = SENetConfig(block_type="bottleneck", layers=(3, 4, 6, 3))
_CFG_101 = SENetConfig(block_type="bottleneck", layers=(3, 4, 23, 3))
_CFG_152 = SENetConfig(block_type="bottleneck", layers=(3, 8, 36, 3))


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_18,
)
def se_resnet_18(pretrained: bool = False, **overrides: object) -> SENet:
    cfg = SENetConfig(**{**_CFG_18.__dict__, **overrides}) if overrides else _CFG_18
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_34,
)
def se_resnet_34(pretrained: bool = False, **overrides: object) -> SENet:
    cfg = SENetConfig(**{**_CFG_34.__dict__, **overrides}) if overrides else _CFG_34
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_50,
)
def se_resnet_50(pretrained: bool = False, **overrides: object) -> SENet:
    cfg = SENetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_101,
)
def se_resnet_101(pretrained: bool = False, **overrides: object) -> SENet:
    cfg = SENetConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_152,
)
def se_resnet_152(pretrained: bool = False, **overrides: object) -> SENet:
    cfg = SENetConfig(**{**_CFG_152.__dict__, **overrides}) if overrides else _CFG_152
    return SENet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_18,
)
def se_resnet_18_cls(
    pretrained: bool = False, **overrides: object
) -> SENetForImageClassification:
    cfg = SENetConfig(**{**_CFG_18.__dict__, **overrides}) if overrides else _CFG_18
    return SENetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_34,
)
def se_resnet_34_cls(
    pretrained: bool = False, **overrides: object
) -> SENetForImageClassification:
    cfg = SENetConfig(**{**_CFG_34.__dict__, **overrides}) if overrides else _CFG_34
    return SENetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_50,
)
def se_resnet_50_cls(
    pretrained: bool = False, **overrides: object
) -> SENetForImageClassification:
    cfg = SENetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return SENetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_101,
)
def se_resnet_101_cls(
    pretrained: bool = False, **overrides: object
) -> SENetForImageClassification:
    cfg = SENetConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return SENetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_152,
)
def se_resnet_152_cls(
    pretrained: bool = False, **overrides: object
) -> SENetForImageClassification:
    cfg = SENetConfig(**{**_CFG_152.__dict__, **overrides}) if overrides else _CFG_152
    return SENetForImageClassification(cfg)
