"""Registry factories for MobileNet v1."""

from lucid.models._registry import register_model
from lucid.models.vision.mobilenet._config import MobileNetV1Config
from lucid.models.vision.mobilenet._model import (
    MobileNetV1,
    MobileNetV1ForImageClassification,
)

_CFG_100 = MobileNetV1Config(width_mult=1.0)
_CFG_075 = MobileNetV1Config(width_mult=0.75)
_CFG_050 = MobileNetV1Config(width_mult=0.5)
_CFG_025 = MobileNetV1Config(width_mult=0.25)


def _b(cfg: MobileNetV1Config, kw: dict[str, object]) -> MobileNetV1:
    return MobileNetV1(MobileNetV1Config(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: MobileNetV1Config, kw: dict[str, object]
) -> MobileNetV1ForImageClassification:
    return MobileNetV1ForImageClassification(
        MobileNetV1Config(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_100,
)
def mobilenet_v1(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    """MobileNet v1 backbone, width_mult=1.0 (Howard et al., 2017)."""
    return _b(_CFG_100, overrides)


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_075,
)
def mobilenet_v1_075(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    return _b(_CFG_075, overrides)


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_050,
)
def mobilenet_v1_050(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    return _b(_CFG_050, overrides)


@register_model(
    task="base",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1,
    default_config=_CFG_025,
)
def mobilenet_v1_025(pretrained: bool = False, **overrides: object) -> MobileNetV1:
    return _b(_CFG_025, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_100,
)
def mobilenet_v1_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV1ForImageClassification:
    """MobileNet v1 classifier, width_mult=1.0."""
    return _c(_CFG_100, overrides)


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_075,
)
def mobilenet_v1_075_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV1ForImageClassification:
    return _c(_CFG_075, overrides)


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_050,
)
def mobilenet_v1_050_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV1ForImageClassification:
    return _c(_CFG_050, overrides)


@register_model(
    task="image-classification",
    family="mobilenet",
    model_type="mobilenet_v1",
    model_class=MobileNetV1ForImageClassification,
    default_config=_CFG_025,
)
def mobilenet_v1_025_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV1ForImageClassification:
    return _c(_CFG_025, overrides)
