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

# Wide ResNet-50-2 / 101-2: 2× width multiplier on bottleneck inner channels.
# Stage output channels remain the same as standard ResNet (256/512/1024/2048).
_CFG_WIDE50 = ResNetConfig(
    block_type="bottleneck", layers=(3, 4, 6, 3), bottleneck_width_mult=2
)
_CFG_WIDE101 = ResNetConfig(
    block_type="bottleneck", layers=(3, 4, 23, 3), bottleneck_width_mult=2
)

# ResNet-200 / ResNet-269: deeper bottleneck variants
_CFG_200 = ResNetConfig(block_type="bottleneck", layers=(3, 24, 36, 3))
_CFG_269 = ResNetConfig(block_type="bottleneck", layers=(3, 30, 48, 8))


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


# ---------------------------------------------------------------------------
# Wide ResNet-50-2
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_WIDE50,
)
def wide_resnet_50(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = (
        ResNetConfig(**{**_CFG_WIDE50.__dict__, **overrides})
        if overrides
        else _CFG_WIDE50
    )
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_WIDE50,
)
def wide_resnet_50_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = (
        ResNetConfig(**{**_CFG_WIDE50.__dict__, **overrides})
        if overrides
        else _CFG_WIDE50
    )
    return ResNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# Wide ResNet-101-2
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_WIDE101,
)
def wide_resnet_101(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = (
        ResNetConfig(**{**_CFG_WIDE101.__dict__, **overrides})
        if overrides
        else _CFG_WIDE101
    )
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_WIDE101,
)
def wide_resnet_101_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = (
        ResNetConfig(**{**_CFG_WIDE101.__dict__, **overrides})
        if overrides
        else _CFG_WIDE101
    )
    return ResNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# ResNet-200
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_200,
)
def resnet_200(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = ResNetConfig(**{**_CFG_200.__dict__, **overrides}) if overrides else _CFG_200
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_200,
)
def resnet_200_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = ResNetConfig(**{**_CFG_200.__dict__, **overrides}) if overrides else _CFG_200
    return ResNetForImageClassification(cfg)


# ---------------------------------------------------------------------------
# ResNet-269
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnet",
    model_type="resnet",
    model_class=ResNet,
    default_config=_CFG_269,
)
def resnet_269(pretrained: bool = False, **overrides: object) -> ResNet:
    cfg = ResNetConfig(**{**_CFG_269.__dict__, **overrides}) if overrides else _CFG_269
    return ResNet(cfg)


@register_model(
    task="image-classification",
    family="resnet",
    model_type="resnet",
    model_class=ResNetForImageClassification,
    default_config=_CFG_269,
)
def resnet_269_cls(
    pretrained: bool = False, **overrides: object
) -> ResNetForImageClassification:
    cfg = ResNetConfig(**{**_CFG_269.__dict__, **overrides}) if overrides else _CFG_269
    return ResNetForImageClassification(cfg)
