"""Registry factories for ResNeSt."""

from lucid.models._registry import register_model
from lucid.models.vision.resnest._config import ResNeStConfig
from lucid.models.vision.resnest._model import (
    ResNeSt,
    ResNeStForImageClassification,
)

_CFG_14 = ResNeStConfig(layers=(1, 1, 1, 1), radix=2, stem_width=32)
_CFG_26 = ResNeStConfig(layers=(2, 2, 2, 2), radix=2, stem_width=32)
_CFG_50 = ResNeStConfig(layers=(3, 4, 6, 3), radix=2)
_CFG_101 = ResNeStConfig(layers=(3, 4, 23, 3), radix=2, stem_width=64)
_CFG_200 = ResNeStConfig(layers=(3, 24, 36, 3), radix=2, stem_width=64)
_CFG_269 = ResNeStConfig(layers=(3, 30, 48, 8), radix=2, stem_width=64)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_14,
)
def resnest_14(pretrained: bool = False, **overrides: object) -> ResNeSt:
    """ResNeSt-14 backbone."""
    cfg = ResNeStConfig(**{**_CFG_14.__dict__, **overrides}) if overrides else _CFG_14
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_26,
)
def resnest_26(pretrained: bool = False, **overrides: object) -> ResNeSt:
    """ResNeSt-26 backbone."""
    cfg = ResNeStConfig(**{**_CFG_26.__dict__, **overrides}) if overrides else _CFG_26
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_50,
)
def resnest_50(pretrained: bool = False, **overrides: object) -> ResNeSt:
    """ResNeSt-50 backbone (Zhang et al., 2020)."""
    cfg = ResNeStConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_101,
)
def resnest_101(pretrained: bool = False, **overrides: object) -> ResNeSt:
    """ResNeSt-101 backbone."""
    cfg = ResNeStConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_200,
)
def resnest_200(pretrained: bool = False, **overrides: object) -> ResNeSt:
    """ResNeSt-200 backbone."""
    cfg = ResNeStConfig(**{**_CFG_200.__dict__, **overrides}) if overrides else _CFG_200
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_269,
)
def resnest_269(pretrained: bool = False, **overrides: object) -> ResNeSt:
    """ResNeSt-269 backbone."""
    cfg = ResNeStConfig(**{**_CFG_269.__dict__, **overrides}) if overrides else _CFG_269
    return ResNeSt(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_14,
)
def resnest_14_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    """ResNeSt-14 classifier."""
    cfg = ResNeStConfig(**{**_CFG_14.__dict__, **overrides}) if overrides else _CFG_14
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_26,
)
def resnest_26_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    """ResNeSt-26 classifier."""
    cfg = ResNeStConfig(**{**_CFG_26.__dict__, **overrides}) if overrides else _CFG_26
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_50,
)
def resnest_50_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    """ResNeSt-50 classifier."""
    cfg = ResNeStConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_101,
)
def resnest_101_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    """ResNeSt-101 classifier."""
    cfg = ResNeStConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_200,
)
def resnest_200_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    """ResNeSt-200 classifier."""
    cfg = ResNeStConfig(**{**_CFG_200.__dict__, **overrides}) if overrides else _CFG_200
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_269,
)
def resnest_269_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    """ResNeSt-269 classifier."""
    cfg = ResNeStConfig(**{**_CFG_269.__dict__, **overrides}) if overrides else _CFG_269
    return ResNeStForImageClassification(cfg)
