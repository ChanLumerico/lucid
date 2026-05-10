"""Registry factories for all ResNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.resnet._config import ResNetConfig
from lucid.models.vision.resnet._model import ResNet, ResNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_18 = ResNetConfig(block_type="basic", layers=(2, 2, 2, 2))
_CFG_34 = ResNetConfig(block_type="basic", layers=(3, 4, 6, 3))
_CFG_50 = ResNetConfig(block_type="bottleneck", layers=(3, 4, 6, 3))
_CFG_101 = ResNetConfig(block_type="bottleneck", layers=(3, 4, 23, 3))
_CFG_152 = ResNetConfig(block_type="bottleneck", layers=(3, 8, 36, 3))


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_18,
)
def resnet_18(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = ResNetConfig(**{**_CFG_18.__dict__, **overrides}) if overrides else _CFG_18
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_34,
)
def resnet_34(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = ResNetConfig(**{**_CFG_34.__dict__, **overrides}) if overrides else _CFG_34
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_50,
)
def resnet_50(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = ResNetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_101,
)
def resnet_101(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = ResNetConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return ResNet(cfg)


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_152,
)
def resnet_152(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = ResNetConfig(**{**_CFG_152.__dict__, **overrides}) if overrides else _CFG_152
    return ResNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_18,
)
def resnet_18_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = ResNetConfig(**{**_CFG_18.__dict__, **overrides}) if overrides else _CFG_18
    return ResNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_34,
)
def resnet_34_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = ResNetConfig(**{**_CFG_34.__dict__, **overrides}) if overrides else _CFG_34
    return ResNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_50,
)
def resnet_50_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = ResNetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return ResNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_101,
)
def resnet_101_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = ResNetConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return ResNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_152,
)
def resnet_152_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = ResNetConfig(**{**_CFG_152.__dict__, **overrides}) if overrides else _CFG_152
    return ResNetForImageClassification(cfg)
