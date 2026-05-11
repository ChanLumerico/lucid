"""Registry factories for ViT variants."""

from lucid.models._registry import register_model
from lucid.models.vision.vit._config import ViTConfig
from lucid.models.vision.vit._model import ViT, ViTForImageClassification

_CFG_B16 = ViTConfig(patch_size=16, dim=768,  depth=12, num_heads=12)
_CFG_B32 = ViTConfig(patch_size=32, dim=768,  depth=12, num_heads=12)
_CFG_L16 = ViTConfig(patch_size=16, dim=1024, depth=24, num_heads=16)
_CFG_L32 = ViTConfig(patch_size=32, dim=1024, depth=24, num_heads=16)
_CFG_H14 = ViTConfig(patch_size=14, dim=1280, depth=32, num_heads=16)


def _b(cfg: ViTConfig, kw: dict[str, object]) -> ViT:
    return ViT(ViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)

def _c(cfg: ViTConfig, kw: dict[str, object]) -> ViTForImageClassification:
    return ViTForImageClassification(ViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


# ── Backbones ─────────────────────────────────────────────────────────────────

@register_model(task="base", family="vit", model_type="vit", model_class=ViT, default_config=_CFG_B16)
def vit_b_16(pretrained: bool = False, **overrides: object) -> ViT:
    """ViT-B/16 backbone (Dosovitskiy et al., 2020)."""
    return _b(_CFG_B16, overrides)

@register_model(task="base", family="vit", model_type="vit", model_class=ViT, default_config=_CFG_B32)
def vit_b_32(pretrained: bool = False, **overrides: object) -> ViT:
    return _b(_CFG_B32, overrides)

@register_model(task="base", family="vit", model_type="vit", model_class=ViT, default_config=_CFG_L16)
def vit_l_16(pretrained: bool = False, **overrides: object) -> ViT:
    return _b(_CFG_L16, overrides)

@register_model(task="base", family="vit", model_type="vit", model_class=ViT, default_config=_CFG_L32)
def vit_l_32(pretrained: bool = False, **overrides: object) -> ViT:
    return _b(_CFG_L32, overrides)

@register_model(task="base", family="vit", model_type="vit", model_class=ViT, default_config=_CFG_H14)
def vit_h_14(pretrained: bool = False, **overrides: object) -> ViT:
    return _b(_CFG_H14, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────

@register_model(task="image-classification", family="vit", model_type="vit", model_class=ViTForImageClassification, default_config=_CFG_B16)
def vit_b_16_cls(pretrained: bool = False, **overrides: object) -> ViTForImageClassification:
    return _c(_CFG_B16, overrides)

@register_model(task="image-classification", family="vit", model_type="vit", model_class=ViTForImageClassification, default_config=_CFG_B32)
def vit_b_32_cls(pretrained: bool = False, **overrides: object) -> ViTForImageClassification:
    return _c(_CFG_B32, overrides)

@register_model(task="image-classification", family="vit", model_type="vit", model_class=ViTForImageClassification, default_config=_CFG_L16)
def vit_l_16_cls(pretrained: bool = False, **overrides: object) -> ViTForImageClassification:
    return _c(_CFG_L16, overrides)

@register_model(task="image-classification", family="vit", model_type="vit", model_class=ViTForImageClassification, default_config=_CFG_L32)
def vit_l_32_cls(pretrained: bool = False, **overrides: object) -> ViTForImageClassification:
    return _c(_CFG_L32, overrides)

@register_model(task="image-classification", family="vit", model_type="vit", model_class=ViTForImageClassification, default_config=_CFG_H14)
def vit_h_14_cls(pretrained: bool = False, **overrides: object) -> ViTForImageClassification:
    return _c(_CFG_H14, overrides)
