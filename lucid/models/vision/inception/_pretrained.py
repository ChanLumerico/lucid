"""Registry factories for Inception v3."""

from lucid.models._registry import register_model
from lucid.models.vision.inception._config import InceptionConfig
from lucid.models.vision.inception._model import (
    InceptionV3,
    InceptionV3ForImageClassification,
)

_CFG = InceptionConfig()
_CFG_NO_AUX = InceptionConfig(aux_logits=False)


@register_model(
    task="base",
    family="inception",
    model_type="inception_v3",
    model_class=InceptionV3,
    default_config=_CFG,
)
def inception_v3(pretrained: bool = False, **overrides: object) -> InceptionV3:
    """Inception v3 backbone (Szegedy et al., 2015)."""
    cfg = InceptionConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionV3(cfg)


@register_model(
    task="image-classification",
    family="inception",
    model_type="inception_v3",
    model_class=InceptionV3ForImageClassification,
    default_config=_CFG,
)
def inception_v3_cls(
    pretrained: bool = False, **overrides: object
) -> InceptionV3ForImageClassification:
    """Inception v3 classifier with auxiliary classifier (Szegedy et al., 2015)."""
    cfg = InceptionConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionV3ForImageClassification(cfg)
