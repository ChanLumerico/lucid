"""Registry factories for EfficientFormer variants."""

from lucid.models._registry import register_model
from lucid.models.vision.efficientformer._config import EfficientFormerConfig
from lucid.models.vision.efficientformer._model import (
    EfficientFormer,
    EfficientFormerForImageClassification,
)

_CFG_L1 = EfficientFormerConfig(
    depths=(3, 2, 6, 4),
    embed_dims=(48, 96, 224, 448),
    mlp_ratios=(4.0, 4.0, 4.0, 4.0),
)

_CFG_L3 = EfficientFormerConfig(
    depths=(4, 4, 12, 6),
    embed_dims=(64, 128, 320, 512),
    mlp_ratios=(4.0, 4.0, 4.0, 4.0),
)

_CFG_L7 = EfficientFormerConfig(
    depths=(6, 6, 18, 8),
    embed_dims=(96, 192, 384, 768),
    mlp_ratios=(4.0, 4.0, 4.0, 4.0),
)


def _b(cfg: EfficientFormerConfig, kw: dict[str, object]) -> EfficientFormer:
    return EfficientFormer(
        EfficientFormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


def _c(
    cfg: EfficientFormerConfig, kw: dict[str, object]
) -> EfficientFormerForImageClassification:
    return EfficientFormerForImageClassification(
        EfficientFormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormer,
    default_config=_CFG_L1,
)
def efficientformer_l1(
    pretrained: bool = False, **overrides: object
) -> EfficientFormer:
    """EfficientFormer-L1 backbone (Li et al., 2022)."""
    return _b(_CFG_L1, overrides)


@register_model(
    task="base",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormer,
    default_config=_CFG_L3,
)
def efficientformer_l3(
    pretrained: bool = False, **overrides: object
) -> EfficientFormer:
    """EfficientFormer-L3 backbone (Li et al., 2022), ~30.9M params."""
    return _b(_CFG_L3, overrides)


@register_model(
    task="base",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormer,
    default_config=_CFG_L7,
)
def efficientformer_l7(
    pretrained: bool = False, **overrides: object
) -> EfficientFormer:
    """EfficientFormer-L7 backbone (Li et al., 2022), ~81.5M params."""
    return _b(_CFG_L7, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormerForImageClassification,
    default_config=_CFG_L1,
)
def efficientformer_l1_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientFormerForImageClassification:
    """EfficientFormer-L1 image classifier (Li et al., 2022)."""
    return _c(_CFG_L1, overrides)


@register_model(
    task="image-classification",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormerForImageClassification,
    default_config=_CFG_L3,
)
def efficientformer_l3_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientFormerForImageClassification:
    """EfficientFormer-L3 image classifier (Li et al., 2022), ~30.9M params."""
    return _c(_CFG_L3, overrides)


@register_model(
    task="image-classification",
    family="efficientformer",
    model_type="efficientformer",
    model_class=EfficientFormerForImageClassification,
    default_config=_CFG_L7,
)
def efficientformer_l7_cls(
    pretrained: bool = False, **overrides: object
) -> EfficientFormerForImageClassification:
    """EfficientFormer-L7 image classifier (Li et al., 2022), ~81.5M params."""
    return _c(_CFG_L7, overrides)
