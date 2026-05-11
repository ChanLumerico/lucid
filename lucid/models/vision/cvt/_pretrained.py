"""Registry factories for CvT variants."""

from lucid.models._registry import register_model
from lucid.models.vision.cvt._config import CvTConfig
from lucid.models.vision.cvt._model import CvT, CvTForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_13 = CvTConfig(
    variant="cvt_13",
    dims=(64, 192, 384),
    depths=(1, 2, 10),
    num_heads=(1, 3, 6),
    embed_strides=(4, 2, 2),
)

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="cvt",
    model_type="cvt",
    model_class=CvT,
    default_config=_CFG_13,
)
def cvt_13(pretrained: bool = False, **overrides: object) -> CvT:
    cfg = CvTConfig(**{**_CFG_13.__dict__, **overrides}) if overrides else _CFG_13
    return CvT(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="cvt",
    model_type="cvt",
    model_class=CvTForImageClassification,
    default_config=_CFG_13,
)
def cvt_13_cls(
    pretrained: bool = False, **overrides: object
) -> CvTForImageClassification:
    cfg = CvTConfig(**{**_CFG_13.__dict__, **overrides}) if overrides else _CFG_13
    return CvTForImageClassification(cfg)
