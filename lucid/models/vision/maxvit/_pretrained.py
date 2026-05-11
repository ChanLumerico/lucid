"""Registry factories for MaxViT variants."""

from lucid.models._registry import register_model
from lucid.models.vision.maxvit._config import MaxViTConfig
from lucid.models.vision.maxvit._model import MaxViT, MaxViTForImageClassification

_CFG_T = MaxViTConfig(
    depths=(2, 2, 5, 2),
    dims=(64, 128, 256, 512),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)

_CFG_S = MaxViTConfig(
    depths=(2, 2, 5, 2),
    dims=(96, 192, 384, 768),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)

_CFG_B = MaxViTConfig(
    depths=(2, 6, 14, 2),
    dims=(96, 192, 384, 768),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)

_CFG_L = MaxViTConfig(
    depths=(2, 6, 14, 2),
    dims=(128, 256, 512, 1024),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)

_CFG_XL = MaxViTConfig(
    depths=(2, 6, 14, 2),
    dims=(192, 384, 768, 1536),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)


def _b(cfg: MaxViTConfig, kw: dict[str, object]) -> MaxViT:
    return MaxViT(MaxViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: MaxViTConfig, kw: dict[str, object]) -> MaxViTForImageClassification:
    return MaxViTForImageClassification(
        MaxViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_T,
)
def maxvit_tiny(pretrained: bool = False, **overrides: object) -> MaxViT:
    """MaxViT-Tiny backbone (Tu et al., 2022)."""
    return _b(_CFG_T, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_S,
)
def maxvit_small(pretrained: bool = False, **overrides: object) -> MaxViT:
    """MaxViT-Small backbone (Tu et al., 2022), ~55.8M params."""
    return _b(_CFG_S, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_B,
)
def maxvit_base(pretrained: bool = False, **overrides: object) -> MaxViT:
    """MaxViT-Base backbone (Tu et al., 2022), ~96.6M params."""
    return _b(_CFG_B, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_L,
)
def maxvit_large(pretrained: bool = False, **overrides: object) -> MaxViT:
    """MaxViT-Large backbone (Tu et al., 2022), ~171.2M params."""
    return _b(_CFG_L, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_XL,
)
def maxvit_xlarge(pretrained: bool = False, **overrides: object) -> MaxViT:
    """MaxViT-XLarge backbone (Tu et al., 2022), ~383.7M params."""
    return _b(_CFG_XL, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_T,
)
def maxvit_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> MaxViTForImageClassification:
    """MaxViT-Tiny image classifier (Tu et al., 2022)."""
    return _c(_CFG_T, overrides)


@register_model(
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_S,
)
def maxvit_small_cls(
    pretrained: bool = False, **overrides: object
) -> MaxViTForImageClassification:
    """MaxViT-Small image classifier (Tu et al., 2022), ~55.8M params."""
    return _c(_CFG_S, overrides)


@register_model(
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_B,
)
def maxvit_base_cls(
    pretrained: bool = False, **overrides: object
) -> MaxViTForImageClassification:
    """MaxViT-Base image classifier (Tu et al., 2022), ~96.6M params."""
    return _c(_CFG_B, overrides)


@register_model(
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_L,
)
def maxvit_large_cls(
    pretrained: bool = False, **overrides: object
) -> MaxViTForImageClassification:
    """MaxViT-Large image classifier (Tu et al., 2022), ~171.2M params."""
    return _c(_CFG_L, overrides)


@register_model(
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_XL,
)
def maxvit_xlarge_cls(
    pretrained: bool = False, **overrides: object
) -> MaxViTForImageClassification:
    """MaxViT-XLarge image classifier (Tu et al., 2022), ~383.7M params."""
    return _c(_CFG_XL, overrides)
