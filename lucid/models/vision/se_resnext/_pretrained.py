"""Registry factories for SE-ResNeXt variants."""

from lucid.models._registry import register_model
from lucid.models.vision.se_resnext._config import SEResNeXtConfig
from lucid.models.vision.se_resnext._model import (
    SEResNeXt,
    SEResNeXtForImageClassification,
)

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_50_32X4D = SEResNeXtConfig(layers=(3, 4, 6, 3),   cardinality=32, base_width=4)
_CFG_101_32X4D = SEResNeXtConfig(layers=(3, 4, 23, 3), cardinality=32, base_width=4)


def _b(cfg: SEResNeXtConfig, kw: dict[str, object]) -> SEResNeXt:
    return SEResNeXt(
        SEResNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


def _c(cfg: SEResNeXtConfig, kw: dict[str, object]) -> SEResNeXtForImageClassification:
    return SEResNeXtForImageClassification(
        SEResNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# Backbones (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="se_resnext",
    model_type="se_resnext",
    model_class=SEResNeXt,
    default_config=_CFG_50_32X4D,
)
def se_resnext_50_32x4d(
    pretrained: bool = False, **overrides: object
) -> SEResNeXt:
    """SE-ResNeXt-50 (32×4d) backbone."""
    return _b(_CFG_50_32X4D, overrides)


@register_model(
    task="base",
    family="se_resnext",
    model_type="se_resnext",
    model_class=SEResNeXt,
    default_config=_CFG_101_32X4D,
)
def se_resnext_101_32x4d(
    pretrained: bool = False, **overrides: object
) -> SEResNeXt:
    """SE-ResNeXt-101 (32×4d) backbone."""
    return _b(_CFG_101_32X4D, overrides)


# ---------------------------------------------------------------------------
# Classifiers (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="se_resnext",
    model_type="se_resnext",
    model_class=SEResNeXtForImageClassification,
    default_config=_CFG_50_32X4D,
)
def se_resnext_50_32x4d_cls(
    pretrained: bool = False, **overrides: object
) -> SEResNeXtForImageClassification:
    """SE-ResNeXt-50 (32×4d) classifier."""
    return _c(_CFG_50_32X4D, overrides)


@register_model(
    task="image-classification",
    family="se_resnext",
    model_type="se_resnext",
    model_class=SEResNeXtForImageClassification,
    default_config=_CFG_101_32X4D,
)
def se_resnext_101_32x4d_cls(
    pretrained: bool = False, **overrides: object
) -> SEResNeXtForImageClassification:
    """SE-ResNeXt-101 (32×4d) classifier."""
    return _c(_CFG_101_32X4D, overrides)
