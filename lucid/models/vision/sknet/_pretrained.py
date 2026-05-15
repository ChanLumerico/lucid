"""Registry factories for all SKNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.sknet._config import SKNetConfig
from lucid.models.vision.sknet._model import SKNet, SKNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

# sk_resnet_18 / sk_resnet_34:
#   basic block (expansion=1, two-SK design), cardinality=1, base_width=64,
#   split_input=False (full-width branches), rd_ratio=0.6 (~3/5).
#   sk_resnet_18: ~24.7M params; sk_resnet_34: ~46.9M params
#   (within ~4% of the reference ~25.6M / ~45.9M targets).
_CFG_SK18 = SKNetConfig(
    layers=(2, 2, 2, 2),
    block_type="basic",
    cardinality=1,
    base_width=64,
    split_input=False,
    rd_ratio=0.6,
)
_CFG_SK34 = SKNetConfig(
    layers=(3, 4, 6, 3),
    block_type="basic",
    cardinality=1,
    base_width=64,
    split_input=False,
    rd_ratio=0.6,
)

# sk_resnet_50 / sk_resnet_101:
#   cardinality=1, base_width=64, split_input=True  →  timm ``skresnet50``
#   25,803,160 parameters for sk_resnet_50_cls (1000-class head)
_CFG_SK50 = SKNetConfig(
    layers=(3, 4, 6, 3), cardinality=1, base_width=64, split_input=True
)
_CFG_SK101 = SKNetConfig(
    layers=(3, 4, 23, 3), cardinality=1, base_width=64, split_input=True
)

# sk_resnext_50_32x4d:
#   cardinality=32, base_width=4, split_input=False, rd_ratio=1/16, rd_divisor=32
#   Equivalent to the SKNet-50 entry in the original paper.  27,479,784 parameters.
_CFG_SK_RX50 = SKNetConfig(
    layers=(3, 4, 6, 3),
    cardinality=32,
    base_width=4,
    split_input=False,
    rd_ratio=1.0 / 16,
    rd_divisor=32,
)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK18,
)
def sk_resnet_18(pretrained: bool = False, **overrides: object) -> SKNet:
    cfg = SKNetConfig(**{**_CFG_SK18.__dict__, **overrides}) if overrides else _CFG_SK18
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK34,
)
def sk_resnet_34(pretrained: bool = False, **overrides: object) -> SKNet:
    cfg = SKNetConfig(**{**_CFG_SK34.__dict__, **overrides}) if overrides else _CFG_SK34
    return SKNet(cfg)


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
    default_config=_CFG_SK18,
)
def sk_resnet_18_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    cfg = SKNetConfig(**{**_CFG_SK18.__dict__, **overrides}) if overrides else _CFG_SK18
    return SKNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK34,
)
def sk_resnet_34_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    cfg = SKNetConfig(**{**_CFG_SK34.__dict__, **overrides}) if overrides else _CFG_SK34
    return SKNetForImageClassification(cfg)


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
