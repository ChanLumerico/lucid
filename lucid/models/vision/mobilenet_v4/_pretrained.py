"""Registry factories for MobileNet v4."""

from lucid.models._registry import register_model
from lucid.models.vision.mobilenet_v4._config import MobileNetV4Config
from lucid.models.vision.mobilenet_v4._model import (
    MobileNetV4,
    MobileNetV4ForImageClassification,
)

_CFG_CONV_SMALL = MobileNetV4Config(variant="conv_small")


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
    cfg = (
        MobileNetV4Config(**{**_CFG_CONV_SMALL.__dict__, **overrides})
        if overrides
        else _CFG_CONV_SMALL
    )
    return MobileNetV4(cfg)


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
    cfg = (
        MobileNetV4Config(**{**_CFG_CONV_SMALL.__dict__, **overrides})
        if overrides
        else _CFG_CONV_SMALL
    )
    return MobileNetV4ForImageClassification(cfg)
