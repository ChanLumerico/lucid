"""Registry factories for Inception v4."""

from lucid.models._registry import register_model
from lucid.models.vision.inception_v4._config import InceptionV4Config
from lucid.models.vision.inception_v4._model import (
    InceptionV4,
    InceptionV4ForImageClassification,
)

_CFG = InceptionV4Config()


@register_model(
    task="base",
    family="inception_v4",
    model_type="inception_v4",
    model_class=InceptionV4,
    default_config=_CFG,
)
def inception_v4(pretrained: bool = False, **overrides: object) -> InceptionV4:
    """Inception v4 backbone (Szegedy et al., 2016)."""
    cfg = InceptionV4Config(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionV4(cfg)


@register_model(
    task="image-classification",
    family="inception_v4",
    model_type="inception_v4",
    model_class=InceptionV4ForImageClassification,
    default_config=_CFG,
)
def inception_v4_cls(
    pretrained: bool = False, **overrides: object
) -> InceptionV4ForImageClassification:
    """Inception v4 classifier (Szegedy et al., 2016)."""
    cfg = InceptionV4Config(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionV4ForImageClassification(cfg)
