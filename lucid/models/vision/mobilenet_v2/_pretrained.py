"""Registry factories for MobileNet v2."""

from lucid.models._registry import register_model
from lucid.models.vision.mobilenet_v2._config import MobileNetV2Config
from lucid.models.vision.mobilenet_v2._model import (
    MobileNetV2,
    MobileNetV2ForImageClassification,
)

_CFG_100 = MobileNetV2Config(width_mult=1.0)
_CFG_075 = MobileNetV2Config(width_mult=0.75)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2,
    default_config=_CFG_100,
)
def mobilenet_v2(pretrained: bool = False, **overrides: object) -> MobileNetV2:
    """MobileNet v2 backbone, width_mult=1.0 (Sandler et al., 2018)."""
    cfg = (
        MobileNetV2Config(**{**_CFG_100.__dict__, **overrides})
        if overrides
        else _CFG_100
    )
    return MobileNetV2(cfg)


@register_model(
    task="base",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2,
    default_config=_CFG_075,
)
def mobilenet_v2_075(pretrained: bool = False, **overrides: object) -> MobileNetV2:
    """MobileNet v2 backbone, width_mult=0.75."""
    cfg = (
        MobileNetV2Config(**{**_CFG_075.__dict__, **overrides})
        if overrides
        else _CFG_075
    )
    return MobileNetV2(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2ForImageClassification,
    default_config=_CFG_100,
)
def mobilenet_v2_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV2ForImageClassification:
    """MobileNet v2 classifier, width_mult=1.0."""
    cfg = (
        MobileNetV2Config(**{**_CFG_100.__dict__, **overrides})
        if overrides
        else _CFG_100
    )
    return MobileNetV2ForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="mobilenet_v2",
    model_type="mobilenet_v2",
    model_class=MobileNetV2ForImageClassification,
    default_config=_CFG_075,
)
def mobilenet_v2_075_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV2ForImageClassification:
    """MobileNet v2 classifier, width_mult=0.75."""
    cfg = (
        MobileNetV2Config(**{**_CFG_075.__dict__, **overrides})
        if overrides
        else _CFG_075
    )
    return MobileNetV2ForImageClassification(cfg)
