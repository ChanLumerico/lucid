"""Registry factories for CSPResNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.cspnet._config import CSPNetConfig
from lucid.models.vision.cspnet._model import CSPNet, CSPNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_50 = CSPNetConfig(layers=(3, 3, 5, 2), channels=(64, 128, 256, 512))

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNet,
    default_config=_CFG_50,
)
def cspresnet_50(pretrained: bool = False, **overrides: object) -> CSPNet:
    cfg = CSPNetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return CSPNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_50,
)
def cspresnet_50_cls(
    pretrained: bool = False, **overrides: object
) -> CSPNetForImageClassification:
    cfg = CSPNetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return CSPNetForImageClassification(cfg)
