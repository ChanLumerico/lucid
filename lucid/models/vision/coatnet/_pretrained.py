"""Registry factories for CoAtNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.coatnet._config import CoAtNetConfig
from lucid.models.vision.coatnet._model import CoAtNet, CoAtNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_0 = CoAtNetConfig(
    variant="coatnet_0",
    blocks_per_stage=(2, 3, 5, 2),
    dims=(96, 192, 384, 768),
    stem_width=64,
    attn_heads=(12, 24),
    mbconv_expand=4,
    head_hidden_size=768,
)

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_0,
)
def coatnet_0(pretrained: bool = False, **overrides: object) -> CoAtNet:
    cfg = CoAtNetConfig(**{**_CFG_0.__dict__, **overrides}) if overrides else _CFG_0
    return CoAtNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_0,
)
def coatnet_0_cls(
    pretrained: bool = False, **overrides: object
) -> CoAtNetForImageClassification:
    cfg = CoAtNetConfig(**{**_CFG_0.__dict__, **overrides}) if overrides else _CFG_0
    return CoAtNetForImageClassification(cfg)
