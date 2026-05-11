"""Registry factories for all SKNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.sknet._config import SKNetConfig
from lucid.models.vision.sknet._model import SKNet, SKNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

# Plain SK-ResNet (cardinality=1 → standard conv inside each SK branch)
_CFG_SK50 = SKNetConfig(layers=(3, 4, 6, 3), cardinality=1, width_per_group=64)
_CFG_SK101 = SKNetConfig(layers=(3, 4, 23, 3), cardinality=1, width_per_group=64)

# SK-ResNeXt (cardinality=32, width_per_group=4 → ResNeXt-50 32×4d style)
_CFG_SK_RX50 = SKNetConfig(layers=(3, 4, 6, 3), cardinality=32, width_per_group=4)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK50,
)
def sk_resnet_50(pretrained: bool = False, **overrides: object) -> SKNet:
    cfg = SKNetConfig(**{**_CFG_SK50.__dict__, **overrides}) if overrides else _CFG_SK50
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK101,
)
def sk_resnet_101(pretrained: bool = False, **overrides: object) -> SKNet:
    cfg = (
        SKNetConfig(**{**_CFG_SK101.__dict__, **overrides}) if overrides else _CFG_SK101
    )
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK_RX50,
)
def sk_resnext_50_32x4d(pretrained: bool = False, **overrides: object) -> SKNet:
    cfg = (
        SKNetConfig(**{**_CFG_SK_RX50.__dict__, **overrides})
        if overrides
        else _CFG_SK_RX50
    )
    return SKNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK50,
)
def sk_resnet_50_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    cfg = SKNetConfig(**{**_CFG_SK50.__dict__, **overrides}) if overrides else _CFG_SK50
    return SKNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK101,
)
def sk_resnet_101_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    cfg = (
        SKNetConfig(**{**_CFG_SK101.__dict__, **overrides}) if overrides else _CFG_SK101
    )
    return SKNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK_RX50,
)
def sk_resnext_50_32x4d_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    cfg = (
        SKNetConfig(**{**_CFG_SK_RX50.__dict__, **overrides})
        if overrides
        else _CFG_SK_RX50
    )
    return SKNetForImageClassification(cfg)
