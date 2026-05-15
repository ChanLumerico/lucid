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

_CFG_21 = CvTConfig(
    variant="cvt_21",
    dims=(64, 192, 384),
    depths=(1, 4, 16),
    num_heads=(1, 3, 6),
    embed_strides=(4, 2, 2),
)

_CFG_W24 = CvTConfig(
    variant="cvt_w24",
    dims=(192, 768, 1024),
    depths=(2, 2, 20),
    num_heads=(3, 12, 16),
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


@register_model(
    task="base",
    family="cvt",
    model_type="cvt",
    model_class=CvT,
    default_config=_CFG_21,
)
def cvt_21(pretrained: bool = False, **overrides: object) -> CvT:
    """CvT-21 backbone (Wu et al., 2021), ~31.6M params."""
    cfg = CvTConfig(**{**_CFG_21.__dict__, **overrides}) if overrides else _CFG_21
    return CvT(cfg)


@register_model(
    task="base",
    family="cvt",
    model_type="cvt",
    model_class=CvT,
    default_config=_CFG_W24,
)
def cvt_w24(pretrained: bool = False, **overrides: object) -> CvT:
    """CvT-W24 wide backbone (Wu et al., 2021), ~277.2M params."""
    cfg = CvTConfig(**{**_CFG_W24.__dict__, **overrides}) if overrides else _CFG_W24
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


@register_model(
    task="image-classification",
    family="cvt",
    model_type="cvt",
    model_class=CvTForImageClassification,
    default_config=_CFG_21,
)
def cvt_21_cls(
    pretrained: bool = False, **overrides: object
) -> CvTForImageClassification:
    """CvT-21 image classifier (Wu et al., 2021), ~31.6M params."""
    cfg = CvTConfig(**{**_CFG_21.__dict__, **overrides}) if overrides else _CFG_21
    return CvTForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="cvt",
    model_type="cvt",
    model_class=CvTForImageClassification,
    default_config=_CFG_W24,
)
def cvt_w24_cls(
    pretrained: bool = False, **overrides: object
) -> CvTForImageClassification:
    """CvT-W24 wide image classifier (Wu et al., 2021), ~277.2M params."""
    cfg = CvTConfig(**{**_CFG_W24.__dict__, **overrides}) if overrides else _CFG_W24
    return CvTForImageClassification(cfg)
