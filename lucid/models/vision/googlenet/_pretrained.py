"""Registry factories for GoogLeNet."""

from lucid.models._registry import register_model
from lucid.models.vision.googlenet._config import GoogLeNetConfig
from lucid.models.vision.googlenet._model import (
    GoogLeNet,
    GoogLeNetForImageClassification,
)

_CFG = GoogLeNetConfig()
_CFG_NO_AUX = GoogLeNetConfig(aux_logits=False)


@register_model(
    task="base",
    family="googlenet",
    model_type="googlenet",
    model_class=GoogLeNet,
    default_config=_CFG,
)
def googlenet(pretrained: bool = False, **overrides: object) -> GoogLeNet:
    """GoogLeNet backbone (Szegedy et al., 2014)."""
    cfg = GoogLeNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return GoogLeNet(cfg)


@register_model(
    task="image-classification",
    family="googlenet",
    model_type="googlenet",
    model_class=GoogLeNetForImageClassification,
    default_config=_CFG,
)
def googlenet_cls(
    pretrained: bool = False, **overrides: object
) -> GoogLeNetForImageClassification:
    """GoogLeNet classifier with auxiliary classifiers (Szegedy et al., 2014)."""
    cfg = GoogLeNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return GoogLeNetForImageClassification(cfg)
