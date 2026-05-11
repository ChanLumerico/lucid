"""Registry factories for EfficientNetV2 variants."""

from lucid.models._registry import register_model
from lucid.models.vision.efficientnet_v2._config import EfficientNetV2Config
from lucid.models.vision.efficientnet_v2._model import (
    EfficientNetV2,
    EfficientNetV2ForImageClassification,
)

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_S = EfficientNetV2Config(variant="small",  dropout=0.2)
_CFG_M = EfficientNetV2Config(variant="medium", dropout=0.3)
_CFG_L = EfficientNetV2Config(variant="large",  dropout=0.4)
_CFG_XL = EfficientNetV2Config(variant="xlarge", dropout=0.4)


def _b(cfg: EfficientNetV2Config, kw: dict[str, object]) -> EfficientNetV2:
    return EfficientNetV2(
        EfficientNetV2Config(**{**cfg.__dict__, **kw}) if kw else cfg
    )


def _c(
    cfg: EfficientNetV2Config, kw: dict[str, object]
) -> EfficientNetV2ForImageClassification:
    return EfficientNetV2ForImageClassification(
        EfficientNetV2Config(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# Backbones (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2,
    default_config=_CFG_S,
)
def efficientnet_v2_small(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2:
    """EfficientNetV2-S backbone (Tan & Le, 2021)."""
    return _b(_CFG_S, overrides)


@register_model(
    task="base",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2,
    default_config=_CFG_M,
)
def efficientnet_v2_medium(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2:
    """EfficientNetV2-M backbone."""
    return _b(_CFG_M, overrides)


@register_model(
    task="base",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2,
    default_config=_CFG_L,
)
def efficientnet_v2_large(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2:
    """EfficientNetV2-L backbone."""
    return _b(_CFG_L, overrides)


@register_model(
    task="base",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2,
    default_config=_CFG_XL,
)
def efficientnet_v2_xlarge(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2:
    """EfficientNetV2-XL backbone."""
    return _b(_CFG_XL, overrides)


# ---------------------------------------------------------------------------
# Classifiers (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2ForImageClassification,
    default_config=_CFG_S,
)
def efficientnet_v2_small_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2ForImageClassification:
    """EfficientNetV2-S classifier."""
    return _c(_CFG_S, overrides)


@register_model(
    task="image-classification",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2ForImageClassification,
    default_config=_CFG_M,
)
def efficientnet_v2_medium_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2ForImageClassification:
    """EfficientNetV2-M classifier."""
    return _c(_CFG_M, overrides)


@register_model(
    task="image-classification",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2ForImageClassification,
    default_config=_CFG_L,
)
def efficientnet_v2_large_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2ForImageClassification:
    """EfficientNetV2-L classifier."""
    return _c(_CFG_L, overrides)


@register_model(
    task="image-classification",
    family="efficientnet_v2",
    model_type="efficientnet_v2",
    model_class=EfficientNetV2ForImageClassification,
    default_config=_CFG_XL,
)
def efficientnet_v2_xlarge_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientNetV2ForImageClassification:
    """EfficientNetV2-XL classifier."""
    return _c(_CFG_XL, overrides)
