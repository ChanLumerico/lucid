"""Registry factories for SqueezeNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.squeezenet._config import SqueezeNetConfig
from lucid.models.vision.squeezenet._model import (
    SqueezeNet,
    SqueezeNetForImageClassification,
)

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_1_0 = SqueezeNetConfig(version="1_0")
_CFG_1_1 = SqueezeNetConfig(version="1_1")


def _b(cfg: SqueezeNetConfig, kw: dict[str, object]) -> SqueezeNet:
    return SqueezeNet(SqueezeNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: SqueezeNetConfig, kw: dict[str, object]
) -> SqueezeNetForImageClassification:
    return SqueezeNetForImageClassification(
        SqueezeNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# Backbones (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="squeezenet",
    model_type="squeezenet",
    model_class=SqueezeNet,
    default_config=_CFG_1_0,
)
def squeezenet_1_0(pretrained: bool = False, **overrides: object) -> SqueezeNet:
    """SqueezeNet 1.0 backbone (Iandola et al., 2016)."""
    return _b(_CFG_1_0, overrides)


@register_model(
    task="base",
    family="squeezenet",
    model_type="squeezenet",
    model_class=SqueezeNet,
    default_config=_CFG_1_1,
)
def squeezenet_1_1(pretrained: bool = False, **overrides: object) -> SqueezeNet:
    """SqueezeNet 1.1 backbone — ~2.4× fewer FLOPs than 1.0."""
    return _b(_CFG_1_1, overrides)


# ---------------------------------------------------------------------------
# Classifiers (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="squeezenet",
    model_type="squeezenet",
    model_class=SqueezeNetForImageClassification,
    default_config=_CFG_1_0,
)
def squeezenet_1_0_cls(
    pretrained: bool = False, **overrides: object
) -> SqueezeNetForImageClassification:
    """SqueezeNet 1.0 classifier."""
    return _c(_CFG_1_0, overrides)


@register_model(
    task="image-classification",
    family="squeezenet",
    model_type="squeezenet",
    model_class=SqueezeNetForImageClassification,
    default_config=_CFG_1_1,
)
def squeezenet_1_1_cls(
    pretrained: bool = False, **overrides: object
) -> SqueezeNetForImageClassification:
    """SqueezeNet 1.1 classifier."""
    return _c(_CFG_1_1, overrides)
