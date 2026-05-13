"""Registry factories for DeiT variants (Touvron et al., 2021)."""

from lucid.models._registry import register_model
from lucid.models.vision.deit._config import DeiTConfig
from lucid.models.vision.deit._model import DeiT, DeiTForImageClassification

# ── Canonical configurations (Table 1 of the paper) ──────────────────────────

_CFG_TINY = DeiTConfig(image_size=224, patch_size=16, dim=192, depth=12, num_heads=3)
_CFG_SMALL = DeiTConfig(image_size=224, patch_size=16, dim=384, depth=12, num_heads=6)
_CFG_BASE = DeiTConfig(image_size=224, patch_size=16, dim=768, depth=12, num_heads=12)
_CFG_BASE_P32 = DeiTConfig(
    image_size=224, patch_size=32, dim=768, depth=12, num_heads=12
)


def _b(cfg: DeiTConfig, kw: dict[str, object]) -> DeiT:
    return DeiT(DeiTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: DeiTConfig, kw: dict[str, object]) -> DeiTForImageClassification:
    return DeiTForImageClassification(
        DeiTConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="deit",
    model_type="deit",
    model_class=DeiT,
    default_config=_CFG_TINY,
)
def deit_tiny(pretrained: bool = False, **overrides: object) -> DeiT:
    """DeiT-Tiny/16 backbone (Touvron et al., 2021)."""
    return _b(_CFG_TINY, overrides)


@register_model(
    task="base",
    family="deit",
    model_type="deit",
    model_class=DeiT,
    default_config=_CFG_SMALL,
)
def deit_small(pretrained: bool = False, **overrides: object) -> DeiT:
    """DeiT-Small/16 backbone (Touvron et al., 2021)."""
    return _b(_CFG_SMALL, overrides)


@register_model(
    task="base",
    family="deit",
    model_type="deit",
    model_class=DeiT,
    default_config=_CFG_BASE,
)
def deit_base(pretrained: bool = False, **overrides: object) -> DeiT:
    """DeiT-Base/16 backbone (Touvron et al., 2021)."""
    return _b(_CFG_BASE, overrides)


@register_model(
    task="base",
    family="deit",
    model_type="deit",
    model_class=DeiT,
    default_config=_CFG_BASE_P32,
)
def deit_base_patch32(pretrained: bool = False, **overrides: object) -> DeiT:
    """DeiT-Base/32 backbone (Touvron et al., 2021)."""
    return _b(_CFG_BASE_P32, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="deit",
    model_type="deit",
    model_class=DeiTForImageClassification,
    default_config=_CFG_TINY,
)
def deit_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> DeiTForImageClassification:
    """DeiT-Tiny/16 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_TINY, overrides)


@register_model(
    task="image-classification",
    family="deit",
    model_type="deit",
    model_class=DeiTForImageClassification,
    default_config=_CFG_SMALL,
)
def deit_small_cls(
    pretrained: bool = False, **overrides: object
) -> DeiTForImageClassification:
    """DeiT-Small/16 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_SMALL, overrides)


@register_model(
    task="image-classification",
    family="deit",
    model_type="deit",
    model_class=DeiTForImageClassification,
    default_config=_CFG_BASE,
)
def deit_base_cls(
    pretrained: bool = False, **overrides: object
) -> DeiTForImageClassification:
    """DeiT-Base/16 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_BASE, overrides)


@register_model(
    task="image-classification",
    family="deit",
    model_type="deit",
    model_class=DeiTForImageClassification,
    default_config=_CFG_BASE_P32,
)
def deit_base_patch32_cls(
    pretrained: bool = False, **overrides: object
) -> DeiTForImageClassification:
    """DeiT-Base/32 image classifier (Touvron et al., 2021)."""
    return _c(_CFG_BASE_P32, overrides)
