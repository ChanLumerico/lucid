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
def maxvit_t(pretrained: bool = False, **overrides: object) -> MaxViT:
    """MaxViT-Tiny backbone (Tu et al., 2022)."""
    return _b(_CFG_T, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_T,
)
def maxvit_t_cls(
    pretrained: bool = False, **overrides: object
) -> MaxViTForImageClassification:
    """MaxViT-Tiny image classifier (Tu et al., 2022)."""
    return _c(_CFG_T, overrides)
