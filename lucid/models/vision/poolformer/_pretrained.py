"""Registry factories for PoolFormer variants."""

from lucid.models._registry import register_model
from lucid.models.vision.poolformer._config import PoolFormerConfig
from lucid.models.vision.poolformer._model import (
    PoolFormer,
    PoolFormerForImageClassification,
)

# Variant configs (paper Table 1)
_CFG_S12 = PoolFormerConfig(layers=(2, 2, 6, 2),   embed_dims=(64, 128, 320, 512))
_CFG_S24 = PoolFormerConfig(layers=(4, 4, 12, 4),  embed_dims=(64, 128, 320, 512))
_CFG_S36 = PoolFormerConfig(layers=(6, 6, 18, 6),  embed_dims=(64, 128, 320, 512))
_CFG_M36 = PoolFormerConfig(layers=(6, 6, 18, 6),  embed_dims=(96, 192, 384, 768))
_CFG_M48 = PoolFormerConfig(layers=(8, 8, 24, 8),  embed_dims=(96, 192, 384, 768))


def _b(cfg: PoolFormerConfig, kw: dict[str, object]) -> PoolFormer:
    return PoolFormer(PoolFormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: PoolFormerConfig, kw: dict[str, object]
) -> PoolFormerForImageClassification:
    return PoolFormerForImageClassification(
        PoolFormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormer,
    default_config=_CFG_S12,
)
def poolformer_s12(pretrained: bool = False, **overrides: object) -> PoolFormer:
    """PoolFormer-S12 backbone (Yu et al., 2022)."""
    return _b(_CFG_S12, overrides)


@register_model(
    task="base",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormer,
    default_config=_CFG_S24,
)
def poolformer_s24(pretrained: bool = False, **overrides: object) -> PoolFormer:
    """PoolFormer-S24 backbone."""
    return _b(_CFG_S24, overrides)


@register_model(
    task="base",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormer,
    default_config=_CFG_S36,
)
def poolformer_s36(pretrained: bool = False, **overrides: object) -> PoolFormer:
    """PoolFormer-S36 backbone."""
    return _b(_CFG_S36, overrides)


@register_model(
    task="base",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormer,
    default_config=_CFG_M36,
)
def poolformer_m36(pretrained: bool = False, **overrides: object) -> PoolFormer:
    """PoolFormer-M36 backbone."""
    return _b(_CFG_M36, overrides)


@register_model(
    task="base",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormer,
    default_config=_CFG_M48,
)
def poolformer_m48(pretrained: bool = False, **overrides: object) -> PoolFormer:
    """PoolFormer-M48 backbone."""
    return _b(_CFG_M48, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormerForImageClassification,
    default_config=_CFG_S12,
)
def poolformer_s12_cls(
    pretrained: bool = False, **overrides: object
) -> PoolFormerForImageClassification:
    """PoolFormer-S12 image classifier."""
    return _c(_CFG_S12, overrides)


@register_model(
    task="image-classification",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormerForImageClassification,
    default_config=_CFG_S24,
)
def poolformer_s24_cls(
    pretrained: bool = False, **overrides: object
) -> PoolFormerForImageClassification:
    """PoolFormer-S24 image classifier."""
    return _c(_CFG_S24, overrides)


@register_model(
    task="image-classification",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormerForImageClassification,
    default_config=_CFG_S36,
)
def poolformer_s36_cls(
    pretrained: bool = False, **overrides: object
) -> PoolFormerForImageClassification:
    """PoolFormer-S36 image classifier."""
    return _c(_CFG_S36, overrides)


@register_model(
    task="image-classification",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormerForImageClassification,
    default_config=_CFG_M36,
)
def poolformer_m36_cls(
    pretrained: bool = False, **overrides: object
) -> PoolFormerForImageClassification:
    """PoolFormer-M36 image classifier."""
    return _c(_CFG_M36, overrides)


@register_model(
    task="image-classification",
    family="poolformer",
    model_type="poolformer",
    model_class=PoolFormerForImageClassification,
    default_config=_CFG_M48,
)
def poolformer_m48_cls(
    pretrained: bool = False, **overrides: object
) -> PoolFormerForImageClassification:
    """PoolFormer-M48 image classifier."""
    return _c(_CFG_M48, overrides)
