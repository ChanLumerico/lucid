"""Registry factories for ZFNet."""

from lucid.models._registry import register_model
from lucid.models.vision.zfnet._config import ZFNetConfig
from lucid.models.vision.zfnet._model import ZFNet, ZFNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical config
# ---------------------------------------------------------------------------

_CFG = ZFNetConfig()


# ---------------------------------------------------------------------------
# Backbone registration (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="zfnet",
    model_type="zfnet",
    model_class=ZFNet,
    default_config=_CFG,
)
def zfnet(pretrained: bool = False, **overrides: object) -> ZFNet:
    cfg = ZFNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return ZFNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registration (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="zfnet",
    model_type="zfnet",
    model_class=ZFNetForImageClassification,
    default_config=_CFG,
)
def zfnet_cls(
    pretrained: bool = False, **overrides: object
) -> ZFNetForImageClassification:
    cfg = ZFNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return ZFNetForImageClassification(cfg)
