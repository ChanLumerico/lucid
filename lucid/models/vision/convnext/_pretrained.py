"""Registry factories for ConvNeXt variants."""

from lucid.models._registry import register_model
from lucid.models.vision.convnext._config import ConvNeXtConfig
from lucid.models.vision.convnext._model import ConvNeXt, ConvNeXtForImageClassification

_CFG_T  = ConvNeXtConfig(depths=(3, 3,  9, 3), dims=(96,  192,  384,  768))
_CFG_S  = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(96,  192,  384,  768))
_CFG_B  = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(128, 256,  512, 1024))
_CFG_L  = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(192, 384,  768, 1536))
_CFG_XL = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(256, 512, 1024, 2048))


def _b(cfg: ConvNeXtConfig, kw: dict[str, object]) -> ConvNeXt:
    return ConvNeXt(ConvNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg)

def _c(cfg: ConvNeXtConfig, kw: dict[str, object]) -> ConvNeXtForImageClassification:
    return ConvNeXtForImageClassification(ConvNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


# ── Backbones ─────────────────────────────────────────────────────────────────

@register_model(task="base", family="convnext", model_type="convnext", model_class=ConvNeXt, default_config=_CFG_T)
def convnext_t(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    """ConvNeXt-T backbone (Liu et al., 2022)."""
    return _b(_CFG_T, overrides)

@register_model(task="base", family="convnext", model_type="convnext", model_class=ConvNeXt, default_config=_CFG_S)
def convnext_s(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    return _b(_CFG_S, overrides)

@register_model(task="base", family="convnext", model_type="convnext", model_class=ConvNeXt, default_config=_CFG_B)
def convnext_b(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    return _b(_CFG_B, overrides)

@register_model(task="base", family="convnext", model_type="convnext", model_class=ConvNeXt, default_config=_CFG_L)
def convnext_l(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    return _b(_CFG_L, overrides)

@register_model(task="base", family="convnext", model_type="convnext", model_class=ConvNeXt, default_config=_CFG_XL)
def convnext_xl(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    return _b(_CFG_XL, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────

@register_model(task="image-classification", family="convnext", model_type="convnext", model_class=ConvNeXtForImageClassification, default_config=_CFG_T)
def convnext_t_cls(pretrained: bool = False, **overrides: object) -> ConvNeXtForImageClassification:
    return _c(_CFG_T, overrides)

@register_model(task="image-classification", family="convnext", model_type="convnext", model_class=ConvNeXtForImageClassification, default_config=_CFG_S)
def convnext_s_cls(pretrained: bool = False, **overrides: object) -> ConvNeXtForImageClassification:
    return _c(_CFG_S, overrides)

@register_model(task="image-classification", family="convnext", model_type="convnext", model_class=ConvNeXtForImageClassification, default_config=_CFG_B)
def convnext_b_cls(pretrained: bool = False, **overrides: object) -> ConvNeXtForImageClassification:
    return _c(_CFG_B, overrides)

@register_model(task="image-classification", family="convnext", model_type="convnext", model_class=ConvNeXtForImageClassification, default_config=_CFG_L)
def convnext_l_cls(pretrained: bool = False, **overrides: object) -> ConvNeXtForImageClassification:
    return _c(_CFG_L, overrides)

@register_model(task="image-classification", family="convnext", model_type="convnext", model_class=ConvNeXtForImageClassification, default_config=_CFG_XL)
def convnext_xl_cls(pretrained: bool = False, **overrides: object) -> ConvNeXtForImageClassification:
    return _c(_CFG_XL, overrides)
