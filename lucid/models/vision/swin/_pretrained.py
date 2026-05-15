"""Registry factories for Swin Transformer variants."""

from lucid.models._registry import register_model
from lucid.models.vision.swin._config import SwinConfig
from lucid.models.vision.swin._model import (
    SwinTransformer,
    SwinTransformerForImageClassification,
)

_CFG_T = SwinConfig(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
_CFG_S = SwinConfig(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
_CFG_B = SwinConfig(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
_CFG_L = SwinConfig(embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))


def _b(cfg: SwinConfig, kw: dict[str, object]) -> SwinTransformer:
    return SwinTransformer(SwinConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: SwinConfig, kw: dict[str, object]) -> SwinTransformerForImageClassification:
    return SwinTransformerForImageClassification(
        SwinConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_T,
)
def swin_tiny(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    """Swin-T backbone (Liu et al., 2021)."""
    return _b(_CFG_T, overrides)


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_S,
)
def swin_small(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    return _b(_CFG_S, overrides)


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_B,
)
def swin_base(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    return _b(_CFG_B, overrides)


@register_model(
    task="base",
    family="swin",
    model_type="swin",
    model_class=SwinTransformer,
    default_config=_CFG_L,
)
def swin_large(pretrained: bool = False, **overrides: object) -> SwinTransformer:
    return _b(_CFG_L, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_T,
)
def swin_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerForImageClassification:
    return _c(_CFG_T, overrides)


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_S,
)
def swin_small_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerForImageClassification:
    return _c(_CFG_S, overrides)


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_B,
)
def swin_base_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerForImageClassification:
    return _c(_CFG_B, overrides)


@register_model(
    task="image-classification",
    family="swin",
    model_type="swin",
    model_class=SwinTransformerForImageClassification,
    default_config=_CFG_L,
)
def swin_large_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerForImageClassification:
    return _c(_CFG_L, overrides)
