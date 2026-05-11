"""Registry factories for AlexNet."""

from lucid.models._registry import register_model
from lucid.models.vision.alexnet._config import AlexNetConfig
from lucid.models.vision.alexnet._model import AlexNet, AlexNetForImageClassification

_CFG = AlexNetConfig()


@register_model(
    task="base",
    family="alexnet",
    model_type="alexnet",
    model_class=AlexNet,
    default_config=_CFG,
)
def alexnet(pretrained: bool = False, **overrides: object) -> AlexNet:
    """AlexNet backbone (Krizhevsky et al., 2012)."""
    cfg = AlexNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return AlexNet(cfg)


@register_model(
    task="image-classification",
    family="alexnet",
    model_type="alexnet",
    model_class=AlexNetForImageClassification,
    default_config=_CFG,
)
def alexnet_cls(pretrained: bool = False, **overrides: object) -> AlexNetForImageClassification:
    """AlexNet classifier (Krizhevsky et al., 2012)."""
    cfg = AlexNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return AlexNetForImageClassification(cfg)
