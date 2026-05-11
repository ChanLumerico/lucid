"""Registry factories for CrossViT variants."""

from lucid.models._registry import register_model
from lucid.models.vision.crossvit._config import CrossViTConfig
from lucid.models.vision.crossvit._model import CrossViT, CrossViTForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_9 = CrossViTConfig(
    depth=3,
    small_dim=128,
    large_dim=256,
    small_patch=12,
    large_patch=16,
    small_heads=4,
    large_heads=4,
    mlp_ratio=3.0,
)

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_9,
)
def crossvit_9(pretrained: bool = False, **overrides: object) -> CrossViT:
    cfg = CrossViTConfig(**{**_CFG_9.__dict__, **overrides}) if overrides else _CFG_9
    return CrossViT(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_9,
)
def crossvit_9_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    cfg = CrossViTConfig(**{**_CFG_9.__dict__, **overrides}) if overrides else _CFG_9
    return CrossViTForImageClassification(cfg)
