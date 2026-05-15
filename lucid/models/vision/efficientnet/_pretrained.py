"""Registry factories for EfficientNet B0–B7."""

from lucid.models._registry import register_model
from lucid.models.vision.efficientnet._config import EfficientNetConfig
from lucid.models.vision.efficientnet._model import (
    EfficientNet,
    EfficientNetForImageClassification,
)

# Compound scaling: (width_mult, depth_mult, dropout)
_CFGS = {
    "b0": EfficientNetConfig(width_mult=1.0, depth_mult=1.0, dropout=0.2),
    "b1": EfficientNetConfig(width_mult=1.0, depth_mult=1.1, dropout=0.2),
    "b2": EfficientNetConfig(width_mult=1.1, depth_mult=1.2, dropout=0.3),
    "b3": EfficientNetConfig(width_mult=1.2, depth_mult=1.4, dropout=0.3),
    "b4": EfficientNetConfig(width_mult=1.4, depth_mult=1.8, dropout=0.4),
    "b5": EfficientNetConfig(width_mult=1.6, depth_mult=2.2, dropout=0.4),
    "b6": EfficientNetConfig(width_mult=1.8, depth_mult=2.6, dropout=0.5),
    "b7": EfficientNetConfig(width_mult=2.0, depth_mult=3.1, dropout=0.5),
}


def _b(key: str, kw: dict[str, object]) -> EfficientNet:
    cfg = _CFGS[key]
    return EfficientNet(EfficientNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(key: str, kw: dict[str, object]) -> EfficientNetForImageClassification:
    cfg = _CFGS[key]
    return EfficientNetForImageClassification(
        EfficientNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b0"],
)
def efficientnet_b0(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b0", overrides)


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b1"],
)
def efficientnet_b1(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b1", overrides)


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b2"],
)
def efficientnet_b2(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b2", overrides)


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b3"],
)
def efficientnet_b3(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b3", overrides)


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b4"],
)
def efficientnet_b4(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b4", overrides)


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b5"],
)
def efficientnet_b5(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b5", overrides)


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b6"],
)
def efficientnet_b6(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b6", overrides)


@register_model(
    task="base",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNet,
    default_config=_CFGS["b7"],
)
def efficientnet_b7(pretrained: bool = False, **overrides: object) -> EfficientNet:
    return _b("b7", overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b0"],
)
def efficientnet_b0_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b0", overrides)


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b1"],
)
def efficientnet_b1_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b1", overrides)


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b2"],
)
def efficientnet_b2_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b2", overrides)


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b3"],
)
def efficientnet_b3_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b3", overrides)


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b4"],
)
def efficientnet_b4_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b4", overrides)


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b5"],
)
def efficientnet_b5_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b5", overrides)


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b6"],
)
def efficientnet_b6_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b6", overrides)


@register_model(
    task="image-classification",
    family="efficientnet",
    model_type="efficientnet",
    model_class=EfficientNetForImageClassification,
    default_config=_CFGS["b7"],
)
def efficientnet_b7_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetForImageClassification:
    return _c("b7", overrides)
