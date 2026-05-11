"""Registry factories for NFNet variants (Brock et al., 2021)."""

from lucid.models._registry import register_model
from lucid.models.vision.nfnet._config import NFNetConfig
from lucid.models.vision.nfnet._model import NFNet, NFNetForImageClassification

# ---------------------------------------------------------------------------
# Default configs — widths and depths from the paper (Table 1).
# ---------------------------------------------------------------------------

# NFNet-F0: smallest variant, ~71 M parameters
_CFG_F0 = NFNetConfig(
    widths=(256, 512, 1536, 1536),
    depths=(1, 2, 6, 3),
    group_size=128,
    alpha=0.2,
    se_ratio=0.5,
    dropout=0.2,
)

# NFNet-F1: depths ×2
_CFG_F1 = NFNetConfig(
    widths=(256, 512, 1536, 1536),
    depths=(2, 4, 12, 6),
    group_size=128,
    alpha=0.2,
    se_ratio=0.5,
    dropout=0.3,
)

# NFNet-F2: depths ×3
_CFG_F2 = NFNetConfig(
    widths=(256, 512, 1536, 1536),
    depths=(3, 6, 18, 9),
    group_size=128,
    alpha=0.2,
    se_ratio=0.5,
    dropout=0.4,
)

# NFNet-F3: depths ×4
_CFG_F3 = NFNetConfig(
    widths=(256, 512, 1536, 1536),
    depths=(4, 8, 24, 12),
    group_size=128,
    alpha=0.2,
    se_ratio=0.5,
    dropout=0.4,
)


def _b(cfg: NFNetConfig, kw: dict[str, object]) -> NFNet:
    return NFNet(NFNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: NFNetConfig, kw: dict[str, object]) -> NFNetForImageClassification:
    return NFNetForImageClassification(
        NFNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNet,
    default_config=_CFG_F0,
)
def nfnet_f0(pretrained: bool = False, **overrides: object) -> NFNet:
    """NFNet-F0 backbone (Brock et al., 2021)."""
    return _b(_CFG_F0, overrides)


@register_model(
    task="base",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNet,
    default_config=_CFG_F1,
)
def nfnet_f1(pretrained: bool = False, **overrides: object) -> NFNet:
    """NFNet-F1 backbone (Brock et al., 2021)."""
    return _b(_CFG_F1, overrides)


@register_model(
    task="base",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNet,
    default_config=_CFG_F2,
)
def nfnet_f2(pretrained: bool = False, **overrides: object) -> NFNet:
    """NFNet-F2 backbone (Brock et al., 2021)."""
    return _b(_CFG_F2, overrides)


@register_model(
    task="base",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNet,
    default_config=_CFG_F3,
)
def nfnet_f3(pretrained: bool = False, **overrides: object) -> NFNet:
    """NFNet-F3 backbone (Brock et al., 2021)."""
    return _b(_CFG_F3, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNetForImageClassification,
    default_config=_CFG_F0,
)
def nfnet_f0_cls(
    pretrained: bool = False, **overrides: object
) -> NFNetForImageClassification:
    """NFNet-F0 image classifier (Brock et al., 2021)."""
    return _c(_CFG_F0, overrides)


@register_model(
    task="image-classification",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNetForImageClassification,
    default_config=_CFG_F1,
)
def nfnet_f1_cls(
    pretrained: bool = False, **overrides: object
) -> NFNetForImageClassification:
    """NFNet-F1 image classifier (Brock et al., 2021)."""
    return _c(_CFG_F1, overrides)


@register_model(
    task="image-classification",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNetForImageClassification,
    default_config=_CFG_F2,
)
def nfnet_f2_cls(
    pretrained: bool = False, **overrides: object
) -> NFNetForImageClassification:
    """NFNet-F2 image classifier (Brock et al., 2021)."""
    return _c(_CFG_F2, overrides)


@register_model(
    task="image-classification",
    family="nfnet",
    model_type="nfnet",
    model_class=NFNetForImageClassification,
    default_config=_CFG_F3,
)
def nfnet_f3_cls(
    pretrained: bool = False, **overrides: object
) -> NFNetForImageClassification:
    """NFNet-F3 image classifier (Brock et al., 2021)."""
    return _c(_CFG_F3, overrides)
