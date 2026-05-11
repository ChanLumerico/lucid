"""Registry factories for Swin Transformer V2 variants."""

from lucid.models._registry import register_model
from lucid.models.vision.swin_v2._config import SwinV2Config
from lucid.models.vision.swin_v2._model import (
    SwinTransformerV2,
    SwinTransformerV2ForImageClassification,
)

_CFG_TINY = SwinV2Config(embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24))
_CFG_SMALL = SwinV2Config(embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24))
_CFG_BASE = SwinV2Config(embed_dim=128, depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32))
_CFG_LARGE = SwinV2Config(embed_dim=192, depths=(2, 2, 18, 2), num_heads=(6, 12, 24, 48))


def _b(cfg: SwinV2Config, kw: dict[str, object]) -> SwinTransformerV2:
    return SwinTransformerV2(SwinV2Config(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: SwinV2Config, kw: dict[str, object]
) -> SwinTransformerV2ForImageClassification:
    return SwinTransformerV2ForImageClassification(
        SwinV2Config(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2,
    default_config=_CFG_TINY,
)
def swin_v2_tiny(pretrained: bool = False, **overrides: object) -> SwinTransformerV2:
    """Swin Transformer V2 tiny backbone (Liu et al., 2022)."""
    return _b(_CFG_TINY, overrides)


@register_model(
    task="base",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2,
    default_config=_CFG_SMALL,
)
def swin_v2_small(pretrained: bool = False, **overrides: object) -> SwinTransformerV2:
    """Swin Transformer V2 small backbone (Liu et al., 2022)."""
    return _b(_CFG_SMALL, overrides)


@register_model(
    task="base",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2,
    default_config=_CFG_BASE,
)
def swin_v2_base(pretrained: bool = False, **overrides: object) -> SwinTransformerV2:
    """Swin Transformer V2 base backbone (Liu et al., 2022)."""
    return _b(_CFG_BASE, overrides)


@register_model(
    task="base",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2,
    default_config=_CFG_LARGE,
)
def swin_v2_large(pretrained: bool = False, **overrides: object) -> SwinTransformerV2:
    """Swin Transformer V2 large backbone (Liu et al., 2022)."""
    return _b(_CFG_LARGE, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2ForImageClassification,
    default_config=_CFG_TINY,
)
def swin_v2_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerV2ForImageClassification:
    """Swin Transformer V2 tiny classifier (Liu et al., 2022)."""
    return _c(_CFG_TINY, overrides)


@register_model(
    task="image-classification",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2ForImageClassification,
    default_config=_CFG_SMALL,
)
def swin_v2_small_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerV2ForImageClassification:
    """Swin Transformer V2 small classifier (Liu et al., 2022)."""
    return _c(_CFG_SMALL, overrides)


@register_model(
    task="image-classification",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2ForImageClassification,
    default_config=_CFG_BASE,
)
def swin_v2_base_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerV2ForImageClassification:
    """Swin Transformer V2 base classifier (Liu et al., 2022)."""
    return _c(_CFG_BASE, overrides)


@register_model(
    task="image-classification",
    family="swin_v2",
    model_type="swin_v2",
    model_class=SwinTransformerV2ForImageClassification,
    default_config=_CFG_LARGE,
)
def swin_v2_large_cls(
    pretrained: bool = False, **overrides: object
) -> SwinTransformerV2ForImageClassification:
    """Swin Transformer V2 large classifier (Liu et al., 2022)."""
    return _c(_CFG_LARGE, overrides)
