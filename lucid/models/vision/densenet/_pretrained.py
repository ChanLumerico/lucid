"""Registry factories for DenseNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.densenet._config import DenseNetConfig
from lucid.models.vision.densenet._model import DenseNet, DenseNetForImageClassification

_CFG_121 = DenseNetConfig(
    block_config=(6, 12, 24, 16), growth_rate=32, num_init_features=64
)
_CFG_169 = DenseNetConfig(
    block_config=(6, 12, 32, 32), growth_rate=32, num_init_features=64
)
_CFG_201 = DenseNetConfig(
    block_config=(6, 12, 48, 32), growth_rate=32, num_init_features=64
)
_CFG_264 = DenseNetConfig(
    block_config=(6, 12, 64, 48), growth_rate=32, num_init_features=64
)


def _b(cfg: DenseNetConfig, kw: dict[str, object]) -> DenseNet:
    return DenseNet(DenseNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: DenseNetConfig, kw: dict[str, object]) -> DenseNetForImageClassification:
    return DenseNetForImageClassification(
        DenseNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_121,
)
def densenet_121(pretrained: bool = False, **overrides: object) -> DenseNet:
    return _b(_CFG_121, overrides)


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_169,
)
def densenet_169(pretrained: bool = False, **overrides: object) -> DenseNet:
    return _b(_CFG_169, overrides)


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_201,
)
def densenet_201(pretrained: bool = False, **overrides: object) -> DenseNet:
    return _b(_CFG_201, overrides)


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_264,
)
def densenet_264(pretrained: bool = False, **overrides: object) -> DenseNet:
    return _b(_CFG_264, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_121,
)
def densenet_121_cls(
    pretrained: bool = False, **overrides: object
) -> DenseNetForImageClassification:
    return _c(_CFG_121, overrides)


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_169,
)
def densenet_169_cls(
    pretrained: bool = False, **overrides: object
) -> DenseNetForImageClassification:
    return _c(_CFG_169, overrides)


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_201,
)
def densenet_201_cls(
    pretrained: bool = False, **overrides: object
) -> DenseNetForImageClassification:
    return _c(_CFG_201, overrides)


@register_model(
    task="image-classification",
    family="densenet",
    model_type="densenet",
    model_class=DenseNetForImageClassification,
    default_config=_CFG_264,
)
def densenet_264_cls(
    pretrained: bool = False, **overrides: object
) -> DenseNetForImageClassification:
    return _c(_CFG_264, overrides)
