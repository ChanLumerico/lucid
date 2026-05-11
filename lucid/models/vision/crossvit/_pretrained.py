"""Registry factories for CrossViT variants."""

from lucid.models._registry import register_model
from lucid.models.vision.crossvit._config import CrossViTConfig
from lucid.models.vision.crossvit._model import CrossViT, CrossViTForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# Paper: Chen et al., 2021 — "CrossViT: Cross-Attention Multi-Scale Vision
# Transformer for Image Classification"
#
# Two-branch ViT: small patch (P_S) + large patch (P_L).
# small_heads = small_dim // 64, large_heads = large_dim // 64 (typical).
# depth = number of CrossViT stages (each stage = self-attn + cross-attn).
# ---------------------------------------------------------------------------

# CrossViT-9  (~8.6 M) — small variant from paper
_CFG_9 = CrossViTConfig(
    depth=3,
    small_dim=128,
    large_dim=256,
    small_patch=12,
    large_patch=16,
    small_heads=4,
    large_heads=4,
    mlp_ratio=3.0,
)

# CrossViT-Tiny (~7.0 M)
_CFG_TINY = CrossViTConfig(
    depth=1,
    small_dim=96,
    large_dim=192,
    small_patch=12,
    large_patch=16,
    small_heads=3,
    large_heads=3,
    mlp_ratio=4.0,
)

# CrossViT-Small (~26.9 M)
_CFG_SMALL = CrossViTConfig(
    depth=4,
    small_dim=192,
    large_dim=384,
    small_patch=12,
    large_patch=16,
    small_heads=6,
    large_heads=6,
    mlp_ratio=4.0,
)

# CrossViT-Base (~105 M) — deeper / wider variant
_CFG_BASE = CrossViTConfig(
    depth=10,
    small_dim=192,
    large_dim=384,
    small_patch=12,
    large_patch=16,
    small_heads=6,
    large_heads=6,
    mlp_ratio=4.0,
)

# CrossViT-15 (~27.5 M)
_CFG_15 = CrossViTConfig(
    depth=3,
    small_dim=192,
    large_dim=384,
    small_patch=12,
    large_patch=16,
    small_heads=6,
    large_heads=6,
    mlp_ratio=3.0,
)

# CrossViT-18 (~43.3 M)
_CFG_18 = CrossViTConfig(
    depth=3,
    small_dim=224,
    large_dim=448,
    small_patch=12,
    large_patch=16,
    small_heads=7,
    large_heads=7,
    mlp_ratio=3.0,
)


def _b(cfg: CrossViTConfig, kw: dict[str, object]) -> CrossViT:
    return CrossViT(CrossViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: CrossViTConfig, kw: dict[str, object]) -> CrossViTForImageClassification:
    return CrossViTForImageClassification(
        CrossViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_9,
)
def crossvit_9(pretrained: bool = False, **overrides: object) -> CrossViT:
    """CrossViT-9 backbone (~8.6 M params)."""
    return _b(_CFG_9, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_TINY,
)
def crossvit_tiny(pretrained: bool = False, **overrides: object) -> CrossViT:
    """CrossViT-Tiny backbone (~7.0 M params)."""
    return _b(_CFG_TINY, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_SMALL,
)
def crossvit_small(pretrained: bool = False, **overrides: object) -> CrossViT:
    """CrossViT-Small backbone (~26.9 M params)."""
    return _b(_CFG_SMALL, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_BASE,
)
def crossvit_base(pretrained: bool = False, **overrides: object) -> CrossViT:
    """CrossViT-Base backbone (~105 M params)."""
    return _b(_CFG_BASE, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_15,
)
def crossvit_15(pretrained: bool = False, **overrides: object) -> CrossViT:
    """CrossViT-15 backbone (~27.5 M params)."""
    return _b(_CFG_15, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_18,
)
def crossvit_18(pretrained: bool = False, **overrides: object) -> CrossViT:
    """CrossViT-18 backbone (~43.3 M params)."""
    return _b(_CFG_18, overrides)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_9,
)
def crossvit_9_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    """CrossViT-9 image classifier (~8.6 M params)."""
    return _c(_CFG_9, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_TINY,
)
def crossvit_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    """CrossViT-Tiny image classifier (~7.0 M params)."""
    return _c(_CFG_TINY, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_SMALL,
)
def crossvit_small_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    """CrossViT-Small image classifier (~26.9 M params)."""
    return _c(_CFG_SMALL, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_BASE,
)
def crossvit_base_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    """CrossViT-Base image classifier (~105 M params)."""
    return _c(_CFG_BASE, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_15,
)
def crossvit_15_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    """CrossViT-15 image classifier (~27.5 M params)."""
    return _c(_CFG_15, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_18,
)
def crossvit_18_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    """CrossViT-18 image classifier (~43.3 M params)."""
    return _c(_CFG_18, overrides)
