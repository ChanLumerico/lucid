"""Registry factories for InceptionNeXt variants."""

from lucid.models._registry import register_model
from lucid.models.vision.inception_next._config import InceptionNeXtConfig
from lucid.models.vision.inception_next._model import (
    InceptionNeXt,
    InceptionNeXtForImageClassification,
)

_CFG_T = InceptionNeXtConfig(
    depths=(3, 3, 9, 3),
    dims=(96, 192, 384, 768),
    band_kernel=11,
)


def _b(cfg: InceptionNeXtConfig, kw: dict[str, object]) -> InceptionNeXt:
    return InceptionNeXt(InceptionNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: InceptionNeXtConfig, kw: dict[str, object]
) -> InceptionNeXtForImageClassification:
    return InceptionNeXtForImageClassification(
        InceptionNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXt,
    default_config=_CFG_T,
)
def inception_next_t(pretrained: bool = False, **overrides: object) -> InceptionNeXt:
    """InceptionNeXt-T backbone (Yu et al., 2023)."""
    return _b(_CFG_T, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXtForImageClassification,
    default_config=_CFG_T,
)
def inception_next_t_cls(
    pretrained: bool = False, **overrides: object
) -> InceptionNeXtForImageClassification:
    """InceptionNeXt-T image classifier (Yu et al., 2023)."""
    return _c(_CFG_T, overrides)
