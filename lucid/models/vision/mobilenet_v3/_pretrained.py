"""Registry factories for MobileNet v3."""

from lucid.models._registry import register_model
from lucid.models.vision.mobilenet_v3._config import MobileNetV3Config
from lucid.models.vision.mobilenet_v3._model import (
    MobileNetV3,
    MobileNetV3ForImageClassification,
)

_CFG_LARGE = MobileNetV3Config(variant="large")
_CFG_SMALL = MobileNetV3Config(variant="small")


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3,
    default_config=_CFG_LARGE,
)
def mobilenet_v3_large(pretrained: bool = False, **overrides: object) -> MobileNetV3:
    """MobileNet v3-Large backbone (Howard et al., 2019)."""
    cfg = (
        MobileNetV3Config(**{**_CFG_LARGE.__dict__, **overrides})
        if overrides
        else _CFG_LARGE
    )
    return MobileNetV3(cfg)


@register_model(
    task="base",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3,
    default_config=_CFG_SMALL,
)
def mobilenet_v3_small(pretrained: bool = False, **overrides: object) -> MobileNetV3:
    """MobileNet v3-Small backbone."""
    cfg = (
        MobileNetV3Config(**{**_CFG_SMALL.__dict__, **overrides})
        if overrides
        else _CFG_SMALL
    )
    return MobileNetV3(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3ForImageClassification,
    default_config=_CFG_LARGE,
)
def mobilenet_v3_large_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV3ForImageClassification:
    """MobileNet v3-Large classifier."""
    cfg = (
        MobileNetV3Config(**{**_CFG_LARGE.__dict__, **overrides})
        if overrides
        else _CFG_LARGE
    )
    return MobileNetV3ForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3ForImageClassification,
    default_config=_CFG_SMALL,
)
def mobilenet_v3_small_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV3ForImageClassification:
    """MobileNet v3-Small classifier."""
    cfg = (
        MobileNetV3Config(**{**_CFG_SMALL.__dict__, **overrides})
        if overrides
        else _CFG_SMALL
    )
    return MobileNetV3ForImageClassification(cfg)
