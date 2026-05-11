"""Registry factories for LeNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.lenet._config import LeNetConfig
from lucid.models.vision.lenet._model import LeNet, LeNetForImageClassification

# Canonical configs
_CFG_5        = LeNetConfig()                               # original (tanh + avg)
_CFG_5_MODERN = LeNetConfig(activation="relu", pooling="max")  # modern convention


# ── Backbone ──────────────────────────────────────────────────────────────────

@register_model(
    task="base",
    family="lenet",
    model_type="lenet",
    model_class=LeNet,
    default_config=_CFG_5,
)
def lenet_5(pretrained: bool = False, **overrides: object) -> LeNet:
    """LeNet-5 backbone — original tanh + avg-pool variant (1998)."""
    cfg = LeNetConfig(**{**_CFG_5.__dict__, **overrides}) if overrides else _CFG_5
    return LeNet(cfg)


@register_model(
    task="base",
    family="lenet",
    model_type="lenet",
    model_class=LeNet,
    default_config=_CFG_5_MODERN,
)
def lenet_5_relu(pretrained: bool = False, **overrides: object) -> LeNet:
    """LeNet-5 backbone — modern ReLU + max-pool variant."""
    cfg = LeNetConfig(**{**_CFG_5_MODERN.__dict__, **overrides}) if overrides else _CFG_5_MODERN
    return LeNet(cfg)


# ── Classifier ────────────────────────────────────────────────────────────────

@register_model(
    task="image-classification",
    family="lenet",
    model_type="lenet",
    model_class=LeNetForImageClassification,
    default_config=_CFG_5,
)
def lenet_5_cls(pretrained: bool = False, **overrides: object) -> LeNetForImageClassification:
    """LeNet-5 classifier — original tanh + avg-pool variant (1998)."""
    cfg = LeNetConfig(**{**_CFG_5.__dict__, **overrides}) if overrides else _CFG_5
    return LeNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="lenet",
    model_type="lenet",
    model_class=LeNetForImageClassification,
    default_config=_CFG_5_MODERN,
)
def lenet_5_relu_cls(pretrained: bool = False, **overrides: object) -> LeNetForImageClassification:
    """LeNet-5 classifier — modern ReLU + max-pool variant."""
    cfg = LeNetConfig(**{**_CFG_5_MODERN.__dict__, **overrides}) if overrides else _CFG_5_MODERN
    return LeNetForImageClassification(cfg)
