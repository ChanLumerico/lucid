"""Registry factories for VGG variants."""

from lucid.models._registry import register_model
from lucid.models.vision.vgg._config import VGGConfig
from lucid.models.vision.vgg._model import VGG, VGGForImageClassification

_CFG_11 = VGGConfig(arch=(1, 1, 2, 2, 2))
_CFG_13 = VGGConfig(arch=(2, 2, 2, 2, 2))
_CFG_16 = VGGConfig(arch=(2, 2, 3, 3, 3))
_CFG_19 = VGGConfig(arch=(2, 2, 4, 4, 4))
_CFG_11_BN = VGGConfig(arch=(1, 1, 2, 2, 2), batch_norm=True)
_CFG_13_BN = VGGConfig(arch=(2, 2, 2, 2, 2), batch_norm=True)
_CFG_16_BN = VGGConfig(arch=(2, 2, 3, 3, 3), batch_norm=True)
_CFG_19_BN = VGGConfig(arch=(2, 2, 4, 4, 4), batch_norm=True)


def _backbone(cfg: VGGConfig, overrides: dict[str, object]) -> VGG:
    if overrides:
        cfg = VGGConfig(**{**cfg.__dict__, **overrides})
    return VGG(cfg)


def _classifier(
    cfg: VGGConfig, overrides: dict[str, object]
) -> VGGForImageClassification:
    if overrides:
        cfg = VGGConfig(**{**cfg.__dict__, **overrides})
    return VGGForImageClassification(cfg)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_11
)
def vgg_11(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_11, overrides)


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_13
)
def vgg_13(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_13, overrides)


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_16
)
def vgg_16(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_16, overrides)


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_19
)
def vgg_19(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_19, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_11_BN,
)
def vgg_11_bn(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_11_BN, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_13_BN,
)
def vgg_13_bn(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_13_BN, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_16_BN,
)
def vgg_16_bn(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_16_BN, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_19_BN,
)
def vgg_19_bn(pretrained: bool = False, **overrides: object) -> VGG:
    return _backbone(_CFG_19_BN, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_11,
)
def vgg_11_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_11, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_13,
)
def vgg_13_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_13, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_16,
)
def vgg_16_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_16, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_19,
)
def vgg_19_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_19, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_11_BN,
)
def vgg_11_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_11_BN, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_13_BN,
)
def vgg_13_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_13_BN, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_16_BN,
)
def vgg_16_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_16_BN, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_19_BN,
)
def vgg_19_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    return _classifier(_CFG_19_BN, overrides)
