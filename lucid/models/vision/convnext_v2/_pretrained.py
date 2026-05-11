"""Registry factories for ConvNeXt V2 variants (Woo et al., 2022)."""

from lucid.models._registry import register_model
from lucid.models.vision.convnext_v2._config import ConvNeXtV2Config
from lucid.models.vision.convnext_v2._model import (
    ConvNeXtV2,
    ConvNeXtV2ForImageClassification,
)

# ── Canonical configurations (Table 1 of the paper) ──────────────────────────

_CFG_ATTO = ConvNeXtV2Config(depths=(2, 2, 6, 2), dims=(40, 80, 160, 320))
_CFG_FEMTO = ConvNeXtV2Config(depths=(2, 2, 6, 2), dims=(48, 96, 192, 384))
_CFG_PICO = ConvNeXtV2Config(depths=(2, 2, 6, 2), dims=(64, 128, 256, 512))
_CFG_NANO = ConvNeXtV2Config(depths=(2, 2, 8, 2), dims=(80, 160, 320, 640))
_CFG_TINY = ConvNeXtV2Config(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
_CFG_SMALL = ConvNeXtV2Config(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768))
_CFG_BASE = ConvNeXtV2Config(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))
_CFG_LARGE = ConvNeXtV2Config(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536))
_CFG_HUGE = ConvNeXtV2Config(depths=(3, 3, 27, 3), dims=(352, 704, 1408, 2816))


def _b(cfg: ConvNeXtV2Config, kw: dict[str, object]) -> ConvNeXtV2:
    return ConvNeXtV2(ConvNeXtV2Config(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: ConvNeXtV2Config, kw: dict[str, object]
) -> ConvNeXtV2ForImageClassification:
    return ConvNeXtV2ForImageClassification(
        ConvNeXtV2Config(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_ATTO,
)
def convnext_v2_atto(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Atto backbone (Woo et al., 2022)."""
    return _b(_CFG_ATTO, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_FEMTO,
)
def convnext_v2_femto(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Femto backbone (Woo et al., 2022)."""
    return _b(_CFG_FEMTO, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_PICO,
)
def convnext_v2_pico(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Pico backbone (Woo et al., 2022)."""
    return _b(_CFG_PICO, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_NANO,
)
def convnext_v2_nano(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Nano backbone (Woo et al., 2022)."""
    return _b(_CFG_NANO, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_TINY,
)
def convnext_v2_tiny(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Tiny backbone (Woo et al., 2022)."""
    return _b(_CFG_TINY, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_SMALL,
)
def convnext_v2_small(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Small backbone (Woo et al., 2022)."""
    return _b(_CFG_SMALL, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_BASE,
)
def convnext_v2_base(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Base backbone (Woo et al., 2022)."""
    return _b(_CFG_BASE, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_LARGE,
)
def convnext_v2_large(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Large backbone (Woo et al., 2022)."""
    return _b(_CFG_LARGE, overrides)


@register_model(
    task="base",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2,
    default_config=_CFG_HUGE,
)
def convnext_v2_huge(pretrained: bool = False, **overrides: object) -> ConvNeXtV2:
    """ConvNeXt V2-Huge backbone (Woo et al., 2022)."""
    return _b(_CFG_HUGE, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_ATTO,
)
def convnext_v2_atto_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Atto image classifier (Woo et al., 2022)."""
    return _c(_CFG_ATTO, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_FEMTO,
)
def convnext_v2_femto_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Femto image classifier (Woo et al., 2022)."""
    return _c(_CFG_FEMTO, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_PICO,
)
def convnext_v2_pico_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Pico image classifier (Woo et al., 2022)."""
    return _c(_CFG_PICO, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_NANO,
)
def convnext_v2_nano_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Nano image classifier (Woo et al., 2022)."""
    return _c(_CFG_NANO, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_TINY,
)
def convnext_v2_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Tiny image classifier (Woo et al., 2022)."""
    return _c(_CFG_TINY, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_SMALL,
)
def convnext_v2_small_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Small image classifier (Woo et al., 2022)."""
    return _c(_CFG_SMALL, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_BASE,
)
def convnext_v2_base_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Base image classifier (Woo et al., 2022)."""
    return _c(_CFG_BASE, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_LARGE,
)
def convnext_v2_large_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Large image classifier (Woo et al., 2022)."""
    return _c(_CFG_LARGE, overrides)


@register_model(
    task="image-classification",
    family="convnext_v2",
    model_type="convnext_v2",
    model_class=ConvNeXtV2ForImageClassification,
    default_config=_CFG_HUGE,
)
def convnext_v2_huge_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtV2ForImageClassification:
    """ConvNeXt V2-Huge image classifier (Woo et al., 2022)."""
    return _c(_CFG_HUGE, overrides)
