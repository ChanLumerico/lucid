"""Registry factories for PVT v2 variants."""

from lucid.models._registry import register_model
from lucid.models.vision.pvt._config import PVTConfig
from lucid.models.vision.pvt._model import PVT, PVTForImageClassification

# ── Canonical configs ─────────────────────────────────────────────────────────

_CFG_B0 = PVTConfig(
    variant="pvt_v2_b0",
    embed_dims=(32, 64, 160, 256),
    depths=(2, 2, 2, 2),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B1 = PVTConfig(
    variant="pvt_v2_b1",
    embed_dims=(64, 128, 320, 512),
    depths=(2, 2, 4, 2),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B2 = PVTConfig(
    variant="pvt_v2_b2",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 4, 6, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B3 = PVTConfig(
    variant="pvt_v2_b3",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 4, 18, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B4 = PVTConfig(
    variant="pvt_v2_b4",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 8, 27, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(8.0, 8.0, 4.0, 4.0),
)

_CFG_B5 = PVTConfig(
    variant="pvt_v2_b5",
    embed_dims=(64, 128, 320, 512),
    depths=(3, 6, 40, 3),
    num_heads=(1, 2, 5, 8),
    sr_ratios=(8, 4, 2, 1),
    mlp_ratios=(4.0, 4.0, 4.0, 4.0),
)

# B1 alias — kept for backwards compatibility
_CFG_TINY = _CFG_B1


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
    default_config=_CFG_B0,
)
def pvt_v2_b0(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT v2-B0 backbone (~3.7 M params)."""
    return _b(_CFG_B0, overrides)


@register_model(
    task="base",
    family="pvt",
    model_type="pvt",
    model_class=PVT,
    default_config=_CFG_B1,
)
def pvt_v2_b1(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT v2-B1 backbone (~14.0 M params)."""
    return _b(_CFG_B1, overrides)


@register_model(
    task="base",
    family="pvt",
    model_type="pvt",
    model_class=PVT,
    default_config=_CFG_B2,
)
def pvt_v2_b2(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT v2-B2 backbone (~25.4 M params)."""
    return _b(_CFG_B2, overrides)


@register_model(
    task="base",
    family="pvt",
    model_type="pvt",
    model_class=PVT,
    default_config=_CFG_B3,
)
def pvt_v2_b3(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT v2-B3 backbone (~45.2 M params)."""
    return _b(_CFG_B3, overrides)


@register_model(
    task="base",
    family="pvt",
    model_type="pvt",
    model_class=PVT,
    default_config=_CFG_B4,
)
def pvt_v2_b4(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT v2-B4 backbone (~62.6 M params)."""
    return _b(_CFG_B4, overrides)


@register_model(
    task="base",
    family="pvt",
    model_type="pvt",
    model_class=PVT,
    default_config=_CFG_B5,
)
def pvt_v2_b5(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT v2-B5 backbone (~82.9 M params)."""
    return _b(_CFG_B5, overrides)


@register_model(
    task="base",
    family="pvt",
    model_type="pvt",
    model_class=PVT,
    default_config=_CFG_TINY,
)
def pvt_tiny(pretrained: bool = False, **overrides: object) -> PVT:
    """PVT-Tiny backbone — alias for pvt_v2_b1 (backwards compat)."""
    return _b(_CFG_TINY, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B0,
)
def pvt_v2_b0_cls(
    pretrained: bool = False, **overrides: object
) -> PVTForImageClassification:
    """PVT v2-B0 image classifier (~3.7 M params)."""
    return _c(_CFG_B0, overrides)


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B1,
)
def pvt_v2_b1_cls(
    pretrained: bool = False, **overrides: object
) -> PVTForImageClassification:
    """PVT v2-B1 image classifier (~14.0 M params)."""
    return _c(_CFG_B1, overrides)


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B2,
)
def pvt_v2_b2_cls(
    pretrained: bool = False, **overrides: object
) -> PVTForImageClassification:
    """PVT v2-B2 image classifier (~25.4 M params)."""
    return _c(_CFG_B2, overrides)


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B3,
)
def pvt_v2_b3_cls(
    pretrained: bool = False, **overrides: object
) -> PVTForImageClassification:
    """PVT v2-B3 image classifier (~45.2 M params)."""
    return _c(_CFG_B3, overrides)


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B4,
)
def pvt_v2_b4_cls(
    pretrained: bool = False, **overrides: object
) -> PVTForImageClassification:
    """PVT v2-B4 image classifier (~62.6 M params)."""
    return _c(_CFG_B4, overrides)


@register_model(
    task="image-classification",
    family="pvt",
    model_type="pvt",
    model_class=PVTForImageClassification,
    default_config=_CFG_B5,
)
def pvt_v2_b5_cls(
    pretrained: bool = False, **overrides: object
) -> PVTForImageClassification:
    """PVT v2-B5 image classifier (~82.9 M params)."""
    return _c(_CFG_B5, overrides)


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
    """PVT-Tiny image classifier — alias for pvt_v2_b1_cls (backwards compat)."""
    return _c(_CFG_TINY, overrides)
