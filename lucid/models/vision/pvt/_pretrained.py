"""Registry factories for PVT variants."""

from lucid.models._registry import register_model
from lucid.models.vision.pvt._config import PVTConfig
from lucid.models.vision.pvt._model import PVT, PVTForImageClassification

_CFG_TINY = PVTConfig(
    variant="pvt_tiny",
    embed_dims=(64, 128, 320, 512),
    depths=(2, 2, 2, 2),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratio=8.0,
)


def _b(cfg: PVTConfig, kw: dict[str, object]) -> PVT:
    return PVT(PVTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: PVTConfig, kw: dict[str, object]) -> PVTForImageClassification:
    return PVTForImageClassification(PVTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="pvt",
    model_type="pvt",
    model_class=PVT,
    default_config=_CFG_TINY,
)
def pvt_tiny(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT-Tiny backbone (Wang et al., 2021)."""
    return _b(_CFG_TINY, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_TINY,
)
def pvt_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> PVTForImageClassification:
    """PVT-Tiny image classifier (Wang et al., 2021)."""
    return _c(_CFG_TINY, overrides)
