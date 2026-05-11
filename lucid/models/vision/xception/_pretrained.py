"""Registry factories for Xception."""

from lucid.models._registry import register_model
from lucid.models.vision.xception._config import XceptionConfig
from lucid.models.vision.xception._model import (
    Xception,
    XceptionForImageClassification,
)

_CFG = XceptionConfig()


@register_model(
    task="base",
    family="xception",
    model_type="xception",
    model_class=Xception,
    default_config=_CFG,
)
def xception(pretrained: bool = False, **overrides: object) -> Xception:
    """Xception backbone (Chollet, 2017)."""
    cfg = XceptionConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return Xception(cfg)


@register_model(
    task="image-classification",
    family="xception",
    model_type="xception",
    model_class=XceptionForImageClassification,
    default_config=_CFG,
)
def xception_cls(
    pretrained: bool = False, **overrides: object
) -> XceptionForImageClassification:
    """Xception classifier (Chollet, 2017)."""
    cfg = XceptionConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return XceptionForImageClassification(cfg)
