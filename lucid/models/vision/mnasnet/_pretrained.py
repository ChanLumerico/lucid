"""Registry factories for MnasNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.mnasnet._config import MnasNetConfig
from lucid.models.vision.mnasnet._model import MnasNet, MnasNetForImageClassification

_CFG_050 = MnasNetConfig(width_mult=0.5)
_CFG_100 = MnasNetConfig(width_mult=1.0)
_CFG_130 = MnasNetConfig(width_mult=1.3)


def _b(cfg: MnasNetConfig, kw: dict[str, object]) -> MnasNet:
    return MnasNet(MnasNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: MnasNetConfig, kw: dict[str, object]) -> MnasNetForImageClassification:
    return MnasNetForImageClassification(
        MnasNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mnasnet",
    model_type="mnasnet",
    model_class=MnasNet,
    default_config=_CFG_050,
)
def mnasnet_050(pretrained: bool = False, **overrides: object) -> MnasNet:
    """MnasNet-0.5 backbone (width_mult=0.5, Tan et al., 2019)."""
    return _b(_CFG_050, overrides)


@register_model(
    task="base",
    family="mnasnet",
    model_type="mnasnet",
    model_class=MnasNet,
    default_config=_CFG_100,
)
def mnasnet_100(pretrained: bool = False, **overrides: object) -> MnasNet:
    """MnasNet-1.0 backbone — MnasNet-A1 baseline (Tan et al., 2019)."""
    return _b(_CFG_100, overrides)


@register_model(
    task="base",
    family="mnasnet",
    model_type="mnasnet",
    model_class=MnasNet,
    default_config=_CFG_130,
)
def mnasnet_130(pretrained: bool = False, **overrides: object) -> MnasNet:
    """MnasNet-1.3 backbone (width_mult=1.3, Tan et al., 2019)."""
    return _b(_CFG_130, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mnasnet",
    model_type="mnasnet",
    model_class=MnasNetForImageClassification,
    default_config=_CFG_050,
)
def mnasnet_050_cls(
    pretrained: bool = False, **overrides: object
) -> MnasNetForImageClassification:
    """MnasNet-0.5 image classifier (width_mult=0.5)."""
    return _c(_CFG_050, overrides)


@register_model(
    task="image-classification",
    family="mnasnet",
    model_type="mnasnet",
    model_class=MnasNetForImageClassification,
    default_config=_CFG_100,
)
def mnasnet_100_cls(
    pretrained: bool = False, **overrides: object
) -> MnasNetForImageClassification:
    """MnasNet-1.0 image classifier — MnasNet-A1 baseline."""
    return _c(_CFG_100, overrides)


@register_model(
    task="image-classification",
    family="mnasnet",
    model_type="mnasnet",
    model_class=MnasNetForImageClassification,
    default_config=_CFG_130,
)
def mnasnet_130_cls(
    pretrained: bool = False, **overrides: object
) -> MnasNetForImageClassification:
    """MnasNet-1.3 image classifier (width_mult=1.3)."""
    return _c(_CFG_130, overrides)
