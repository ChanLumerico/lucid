"""Registry factories for all ResNeXt variants."""

from lucid.models._registry import register_model
from lucid.models.vision.resnext._config import ResNeXtConfig
from lucid.models.vision.resnext._model import ResNeXt, ResNeXtForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_50_32x4d = ResNeXtConfig(layers=(3, 4, 6, 3), cardinality=32, width_per_group=4)
_CFG_101_32x4d = ResNeXtConfig(layers=(3, 4, 23, 3), cardinality=32, width_per_group=4)
_CFG_101_32x8d = ResNeXtConfig(layers=(3, 4, 23, 3), cardinality=32, width_per_group=8)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_50_32x4d,
)
def resnext_50_32x4d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    cfg = (
        ResNeXtConfig(**{**_CFG_50_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_50_32x4d
    )
    return ResNeXt(cfg)


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_101_32x4d,
)
def resnext_101_32x4d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x4d
    )
    return ResNeXt(cfg)


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_101_32x8d,
)
def resnext_101_32x8d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x8d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x8d
    )
    return ResNeXt(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_50_32x4d,
)
def resnext_50_32x4d_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeXtForImageClassification:
    cfg = (
        ResNeXtConfig(**{**_CFG_50_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_50_32x4d
    )
    return ResNeXtForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_101_32x4d,
)
def resnext_101_32x4d_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeXtForImageClassification:
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x4d
    )
    return ResNeXtForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_101_32x8d,
)
def resnext_101_32x8d_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeXtForImageClassification:
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x8d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x8d
    )
    return ResNeXtForImageClassification(cfg)
