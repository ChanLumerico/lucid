"""Registry factories for MobileNet v4."""

from lucid.models._registry import register_model
from lucid.models.vision.mobilenet_v4._config import MobileNetV4Config
from lucid.models.vision.mobilenet_v4._model import (
    MobileNetV4,
    MobileNetV4ForImageClassification,
)

_CFG_CONV_SMALL = MobileNetV4Config(variant="conv_small")
_CFG_CONV_MEDIUM = MobileNetV4Config(variant="conv_medium")
_CFG_CONV_LARGE = MobileNetV4Config(variant="conv_large")


def _b(cfg: MobileNetV4Config, kw: dict[str, object]) -> MobileNetV4:
    return MobileNetV4(MobileNetV4Config(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: MobileNetV4Config, kw: dict[str, object]
) -> MobileNetV4ForImageClassification:
    return MobileNetV4ForImageClassification(
        MobileNetV4Config(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_CONV_SMALL,
)
def mobilenet_v4_conv_small(
    pretrained: bool = False, **overrides: object
) -> MobileNetV4:
    """MobileNet v4 Conv-Small backbone (Qin et al., 2024)."""
    return _b(_CFG_CONV_SMALL, overrides)


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_CONV_MEDIUM,
)
def mobilenet_v4_conv_medium(
    pretrained: bool = False, **overrides: object
) -> MobileNetV4:
    """MobileNet v4 Conv-Medium backbone (~9.7 M params)."""
    return _b(_CFG_CONV_MEDIUM, overrides)


@register_model(
    task="base",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4,
    default_config=_CFG_CONV_LARGE,
)
def mobilenet_v4_conv_large(
    pretrained: bool = False, **overrides: object
) -> MobileNetV4:
    """MobileNet v4 Conv-Large backbone (~32.6 M params)."""
    return _b(_CFG_CONV_LARGE, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_CONV_SMALL,
)
def mobilenet_v4_conv_small_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV4ForImageClassification:
    """MobileNet v4 Conv-Small classifier."""
    return _c(_CFG_CONV_SMALL, overrides)


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_CONV_MEDIUM,
)
def mobilenet_v4_conv_medium_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV4ForImageClassification:
    """MobileNet v4 Conv-Medium classifier (~9.7 M params)."""
    return _c(_CFG_CONV_MEDIUM, overrides)


@register_model(
    task="image-classification",
    family="mobilenet_v4",
    model_type="mobilenet_v4",
    model_class=MobileNetV4ForImageClassification,
    default_config=_CFG_CONV_LARGE,
)
def mobilenet_v4_conv_large_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV4ForImageClassification:
    """MobileNet v4 Conv-Large classifier (~32.6 M params)."""
    return _c(_CFG_CONV_LARGE, overrides)
