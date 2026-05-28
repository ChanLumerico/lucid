"""Registry factories for the paper-cited CrossViT variants.

All six configs from Chen et al., ICCV 2021, Table 2.  Per-variant
hyperparameters are sourced from the paper directly and cross-checked
against ``timm.models.crossvit`` so the converted timm checkpoints
load with a direct ``blocks → stages`` key rename.
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.crossvit._config import CrossViTConfig
from lucid.models.vision.crossvit._model import (
    CrossViT,
    CrossViTForImageClassification,
)
from lucid.models.vision.crossvit._weights import (
    CrossViT9Weights,
    CrossViT15Weights,
    CrossViT18Weights,
    CrossViTBaseWeights,
    CrossViTSmallWeights,
    CrossViTTinyWeights,
)

# ---------------------------------------------------------------------------
# Paper Table 2 configurations.
#
# Every variant shares the same skeleton (K=3 MultiScaleBlocks, patch sizes
# (12, 16), dual-input scale (240, 224)); only ``embed_dims`` / ``depths`` /
# ``num_heads`` / ``mlp_ratio`` change.
# ---------------------------------------------------------------------------

_DEPTH_TINY = ((1, 4, 0), (1, 4, 0), (1, 4, 0))
_DEPTH_SMALL = ((1, 4, 0), (1, 4, 0), (1, 4, 0))
_DEPTH_BASE = ((1, 4, 0), (1, 4, 0), (1, 4, 0))
_DEPTH_9 = ((1, 3, 0), (1, 3, 0), (1, 3, 0))
_DEPTH_15 = ((1, 5, 0), (1, 5, 0), (1, 5, 0))
_DEPTH_18 = ((1, 6, 0), (1, 6, 0), (1, 6, 0))

_MLP_BCT = (4.0, 4.0, 1.0)  # tiny / small / base
_MLP_9_15_18 = (3.0, 3.0, 1.0)  # 9 / 15 / 18

_CFG_TINY = CrossViTConfig(
    embed_dims=(96, 192),
    depths=_DEPTH_TINY,
    num_heads=(3, 3),
    mlp_ratio=_MLP_BCT,
)
_CFG_SMALL = CrossViTConfig(
    embed_dims=(192, 384),
    depths=_DEPTH_SMALL,
    num_heads=(6, 6),
    mlp_ratio=_MLP_BCT,
)
_CFG_BASE = CrossViTConfig(
    embed_dims=(384, 768),
    depths=_DEPTH_BASE,
    num_heads=(12, 12),
    mlp_ratio=_MLP_BCT,
)
_CFG_9 = CrossViTConfig(
    embed_dims=(128, 256),
    depths=_DEPTH_9,
    num_heads=(4, 4),
    mlp_ratio=_MLP_9_15_18,
)
_CFG_15 = CrossViTConfig(
    embed_dims=(192, 384),
    depths=_DEPTH_15,
    num_heads=(6, 6),
    mlp_ratio=_MLP_9_15_18,
)
_CFG_18 = CrossViTConfig(
    embed_dims=(224, 448),
    depths=_DEPTH_18,
    num_heads=(7, 7),
    mlp_ratio=_MLP_9_15_18,
)


def _b(cfg: CrossViTConfig, kw: dict[str, object]) -> CrossViT:
    return CrossViT(CrossViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: CrossViTConfig, kw: dict[str, object]) -> CrossViTForImageClassification:
    return CrossViTForImageClassification(
        CrossViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# Backbone factories (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base", family="crossvit", model_type="crossvit",
    model_class=CrossViT, default_config=_CFG_TINY, params=7_000_000,
)
def crossvit_tiny(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-Ti backbone — ``embed_dims=(96, 192)``, depths
    ``((1, 4, 0))×3``, 3 heads.  ~7M params (paper Table 2)."""
    return _b(_CFG_TINY, overrides)


@register_model(
    task="base", family="crossvit", model_type="crossvit",
    model_class=CrossViT, default_config=_CFG_SMALL, params=26_700_000,
)
def crossvit_small(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-S backbone — ``embed_dims=(192, 384)``, depths
    ``((1, 4, 0))×3``, 6 heads.  ~26.7M params (paper Table 2)."""
    return _b(_CFG_SMALL, overrides)


@register_model(
    task="base", family="crossvit", model_type="crossvit",
    model_class=CrossViT, default_config=_CFG_BASE, params=105_000_000,
)
def crossvit_base(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-B backbone — ``embed_dims=(384, 768)``, depths
    ``((1, 4, 0))×3``, 12 heads.  ~105M params (paper Table 2)."""
    return _b(_CFG_BASE, overrides)


@register_model(
    task="base", family="crossvit", model_type="crossvit",
    model_class=CrossViT, default_config=_CFG_9, params=8_600_000,
)
def crossvit_9(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-9 backbone — ``embed_dims=(128, 256)``, depths
    ``((1, 3, 0))×3``, 4 heads.  ~8.6M params (paper Table 2)."""
    return _b(_CFG_9, overrides)


@register_model(
    task="base", family="crossvit", model_type="crossvit",
    model_class=CrossViT, default_config=_CFG_15, params=27_400_000,
)
def crossvit_15(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-15 backbone — ``embed_dims=(192, 384)``, depths
    ``((1, 5, 0))×3``, 6 heads.  ~27.4M params (paper Table 2)."""
    return _b(_CFG_15, overrides)


@register_model(
    task="base", family="crossvit", model_type="crossvit",
    model_class=CrossViT, default_config=_CFG_18, params=43_300_000,
)
def crossvit_18(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-18 backbone — ``embed_dims=(224, 448)``, depths
    ``((1, 6, 0))×3``, 7 heads.  ~43.3M params (paper Table 2)."""
    return _b(_CFG_18, overrides)


# ---------------------------------------------------------------------------
# Classification factories with pretrained-weight wiring (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_TINY,
    params=7_000_000,
)
def crossvit_tiny_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViTTinyWeights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-Ti image classifier (Chen et al., ICCV 2021).

    Two-branch ViT with embedding dims ``(96, 192)``, per-stage depths
    ``((1, 4, 0)) × 3``, and 3 attention heads per branch.  ~7M
    parameters; paper Table 2 reports **72.6%** ImageNet-1k top-1 at
    240×240.
    """
    entry = weights_mod.resolve_weights(CrossViTTinyWeights, pretrained, weights)
    model = _c(_CFG_TINY, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_tiny_cls")
    return model


@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_SMALL,
    params=26_700_000,
)
def crossvit_small_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViTSmallWeights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-S image classifier — ``embed_dims=(192, 384)``, depths
    ``((1, 4, 0)) × 3``, 6 heads per branch.  ~26.7M params; paper
    Table 2 reports **81.0%** ImageNet-1k top-1 at 240×240."""
    entry = weights_mod.resolve_weights(CrossViTSmallWeights, pretrained, weights)
    model = _c(_CFG_SMALL, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_small_cls")
    return model


@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_BASE,
    params=105_000_000,
)
def crossvit_base_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViTBaseWeights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-B image classifier — ``embed_dims=(384, 768)``, depths
    ``((1, 4, 0)) × 3``, 12 heads per branch.  ~105M params; paper
    Table 2 reports **82.2%** ImageNet-1k top-1 at 240×240."""
    entry = weights_mod.resolve_weights(CrossViTBaseWeights, pretrained, weights)
    model = _c(_CFG_BASE, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_base_cls")
    return model


@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_9,
    params=8_600_000,
)
def crossvit_9_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViT9Weights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-9 image classifier — ``embed_dims=(128, 256)``, depths
    ``((1, 3, 0)) × 3``, 4 heads per branch.  ~8.6M params; paper
    Table 2 reports **73.9%** ImageNet-1k top-1 at 240×240."""
    entry = weights_mod.resolve_weights(CrossViT9Weights, pretrained, weights)
    model = _c(_CFG_9, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_9_cls")
    return model


@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_15,
    params=27_400_000,
)
def crossvit_15_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViT15Weights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-15 image classifier — ``embed_dims=(192, 384)``, depths
    ``((1, 5, 0)) × 3``, 6 heads per branch.  ~27.4M params; paper
    Table 2 reports **81.5%** ImageNet-1k top-1 at 240×240."""
    entry = weights_mod.resolve_weights(CrossViT15Weights, pretrained, weights)
    model = _c(_CFG_15, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_15_cls")
    return model


@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_18,
    params=43_300_000,
)
def crossvit_18_cls(
    pretrained: bool | str = False,
    *,
    weights: CrossViT18Weights | None = None,
    **overrides: object,
) -> CrossViTForImageClassification:
    r"""CrossViT-18 image classifier — ``embed_dims=(224, 448)``, depths
    ``((1, 6, 0)) × 3``, 7 heads per branch.  ~43.3M params; paper
    Table 2 reports **82.5%** ImageNet-1k top-1 at 240×240."""
    entry = weights_mod.resolve_weights(CrossViT18Weights, pretrained, weights)
    model = _c(_CFG_18, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="crossvit_18_cls")
    return model
