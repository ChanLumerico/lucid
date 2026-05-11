"""Registry factories for CaiT variants (Touvron et al., 2021)."""

from lucid.models._registry import register_model
from lucid.models.vision.cait._config import CaiTConfig
from lucid.models.vision.cait._model import CaiT, CaiTForImageClassification

# ---------------------------------------------------------------------------
# Default configs — from Table 1 of Touvron et al. (2021).
# ---------------------------------------------------------------------------

_CFG_XXS24 = CaiTConfig(dim=192, depth=24, num_heads=4, layer_scale_init=1e-5)
_CFG_XXS36 = CaiTConfig(dim=192, depth=36, num_heads=4, layer_scale_init=1e-5)
_CFG_XS24 = CaiTConfig(dim=288, depth=24, num_heads=6, layer_scale_init=1e-5)
_CFG_S24 = CaiTConfig(dim=384, depth=24, num_heads=8, layer_scale_init=1e-5)
_CFG_S36 = CaiTConfig(dim=384, depth=36, num_heads=8, layer_scale_init=1e-5)
_CFG_M36 = CaiTConfig(dim=768, depth=36, num_heads=16, layer_scale_init=1e-6)
_CFG_M48 = CaiTConfig(dim=768, depth=48, num_heads=16, layer_scale_init=1e-6)


def _b(cfg: CaiTConfig, kw: dict[str, object]) -> CaiT:
    return CaiT(CaiTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: CaiTConfig, kw: dict[str, object]) -> CaiTForImageClassification:
    return CaiTForImageClassification(
        CaiTConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="cait",
    model_type="cait",
    model_class=CaiT,
    default_config=_CFG_XXS24,
)
def cait_xxsmall_24(pretrained: bool = False, **overrides: object) -> CaiT:
    """CaiT-XXS/24 backbone (Touvron et al., 2021) — dim=192, depth=24, heads=4."""
    return _b(_CFG_XXS24, overrides)


@register_model(
    task="base",
    family="cait",
    model_type="cait",
    model_class=CaiT,
    default_config=_CFG_XXS36,
)
def cait_xxsmall_36(pretrained: bool = False, **overrides: object) -> CaiT:
    """CaiT-XXS/36 backbone (Touvron et al., 2021) — dim=192, depth=36, heads=4."""
    return _b(_CFG_XXS36, overrides)


@register_model(
    task="base",
    family="cait",
    model_type="cait",
    model_class=CaiT,
    default_config=_CFG_XS24,
)
def cait_xsmall_24(pretrained: bool = False, **overrides: object) -> CaiT:
    """CaiT-XS/24 backbone (Touvron et al., 2021) — dim=288, depth=24, heads=6."""
    return _b(_CFG_XS24, overrides)


@register_model(
    task="base",
    family="cait",
    model_type="cait",
    model_class=CaiT,
    default_config=_CFG_S24,
)
def cait_small_24(pretrained: bool = False, **overrides: object) -> CaiT:
    """CaiT-S/24 backbone (Touvron et al., 2021) — dim=384, depth=24, heads=8."""
    return _b(_CFG_S24, overrides)


@register_model(
    task="base",
    family="cait",
    model_type="cait",
    model_class=CaiT,
    default_config=_CFG_S36,
)
def cait_small_36(pretrained: bool = False, **overrides: object) -> CaiT:
    """CaiT-S/36 backbone (Touvron et al., 2021) — dim=384, depth=36, heads=8."""
    return _b(_CFG_S36, overrides)


@register_model(
    task="base",
    family="cait",
    model_type="cait",
    model_class=CaiT,
    default_config=_CFG_M36,
)
def cait_medium_36(pretrained: bool = False, **overrides: object) -> CaiT:
    """CaiT-M/36 backbone (Touvron et al., 2021) — dim=768, depth=36, heads=16."""
    return _b(_CFG_M36, overrides)


@register_model(
    task="base",
    family="cait",
    model_type="cait",
    model_class=CaiT,
    default_config=_CFG_M48,
)
def cait_medium_48(pretrained: bool = False, **overrides: object) -> CaiT:
    """CaiT-M/48 backbone (Touvron et al., 2021) — dim=768, depth=48, heads=16."""
    return _b(_CFG_M48, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="cait",
    model_type="cait",
    model_class=CaiTForImageClassification,
    default_config=_CFG_XXS24,
)
def cait_xxsmall_24_cls(
    pretrained: bool = False, **overrides: object
) -> CaiTForImageClassification:
    """CaiT-XXS/24 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_XXS24, overrides)


@register_model(
    task="image-classification",
    family="cait",
    model_type="cait",
    model_class=CaiTForImageClassification,
    default_config=_CFG_XXS36,
)
def cait_xxsmall_36_cls(
    pretrained: bool = False, **overrides: object
) -> CaiTForImageClassification:
    """CaiT-XXS/36 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_XXS36, overrides)


@register_model(
    task="image-classification",
    family="cait",
    model_type="cait",
    model_class=CaiTForImageClassification,
    default_config=_CFG_XS24,
)
def cait_xsmall_24_cls(
    pretrained: bool = False, **overrides: object
) -> CaiTForImageClassification:
    """CaiT-XS/24 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_XS24, overrides)


@register_model(
    task="image-classification",
    family="cait",
    model_type="cait",
    model_class=CaiTForImageClassification,
    default_config=_CFG_S24,
)
def cait_small_24_cls(
    pretrained: bool = False, **overrides: object
) -> CaiTForImageClassification:
    """CaiT-S/24 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_S24, overrides)


@register_model(
    task="image-classification",
    family="cait",
    model_type="cait",
    model_class=CaiTForImageClassification,
    default_config=_CFG_S36,
)
def cait_small_36_cls(
    pretrained: bool = False, **overrides: object
) -> CaiTForImageClassification:
    """CaiT-S/36 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_S36, overrides)


@register_model(
    task="image-classification",
    family="cait",
    model_type="cait",
    model_class=CaiTForImageClassification,
    default_config=_CFG_M36,
)
def cait_medium_36_cls(
    pretrained: bool = False, **overrides: object
) -> CaiTForImageClassification:
    """CaiT-M/36 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_M36, overrides)


@register_model(
    task="image-classification",
    family="cait",
    model_type="cait",
    model_class=CaiTForImageClassification,
    default_config=_CFG_M48,
)
def cait_medium_48_cls(
    pretrained: bool = False, **overrides: object
) -> CaiTForImageClassification:
    """CaiT-M/48 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_M48, overrides)
