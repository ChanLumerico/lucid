"""Registry factories for the three paper-cited CSPNet variants.

Wang et al., CVPRW 2020.  Three architectures ship: ``cspresnet_50``,
``cspresnext_50``, ``cspdarknet_53``.  Hyperparameters lifted from
``timm.models.cspnet`` to keep state-dict compatibility (the converter
in ``tools/convert_weights/cspnet.py`` is a single ``map_key`` for all
three variants).
"""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.cspnet._config import CSPNetConfig
from lucid.models.vision.cspnet._model import CSPNet, CSPNetForImageClassification
from lucid.models.vision.cspnet._weights import (
    CSPDarknet53Weights,
    CSPResNet50Weights,
    CSPResNeXt50Weights,
)

# ---------------------------------------------------------------------------
# Per-variant configs (timm ``model_cfgs`` values, paper-faithful).
# ---------------------------------------------------------------------------

_CFG_CSPRESNET_50 = CSPNetConfig(
    stem_out_chs=64,
    stem_kernel=7,
    stem_stride=2,
    stem_pool="max",
    depths=(3, 3, 5, 2),
    out_chs=(128, 256, 512, 1024),
    strides=(1, 2, 2, 2),
    groups=(1, 1, 1, 1),
    expand_ratio=(2.0, 2.0, 2.0, 2.0),
    bottle_ratio=(0.5, 0.5, 0.5, 0.5),
    block_ratio=(1.0, 1.0, 1.0, 1.0),
    cross_linear=(True, True, True, True),
    down_growth=(False, False, False, False),
    block_type=("bottle", "bottle", "bottle", "bottle"),
)

_CFG_CSPRESNEXT_50 = CSPNetConfig(
    stem_out_chs=64,
    stem_kernel=7,
    stem_stride=2,
    stem_pool="max",
    depths=(3, 3, 5, 2),
    out_chs=(256, 512, 1024, 2048),
    strides=(1, 2, 2, 2),
    groups=(32, 32, 32, 32),
    expand_ratio=(1.0, 1.0, 1.0, 1.0),
    bottle_ratio=(1.0, 1.0, 1.0, 1.0),
    block_ratio=(0.5, 0.5, 0.5, 0.5),
    cross_linear=(True, True, True, True),
    down_growth=(False, False, False, False),
    block_type=("bottle", "bottle", "bottle", "bottle"),
)

_CFG_CSPDARKNET_53 = CSPNetConfig(
    stem_out_chs=32,
    stem_kernel=3,
    stem_stride=1,
    stem_pool="",
    depths=(1, 2, 8, 8, 4),
    out_chs=(64, 128, 256, 512, 1024),
    strides=(2, 2, 2, 2, 2),
    groups=(1, 1, 1, 1, 1),
    expand_ratio=(2.0, 1.0, 1.0, 1.0, 1.0),
    bottle_ratio=(0.5, 1.0, 1.0, 1.0, 1.0),
    block_ratio=(1.0, 0.5, 0.5, 0.5, 0.5),
    cross_linear=(False, False, False, False, False),
    down_growth=(True, True, True, True, True),
    block_type=("dark", "dark", "dark", "dark", "dark"),
)


def _b(cfg: CSPNetConfig, kw: dict[str, object]) -> CSPNet:
    return CSPNet(CSPNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: CSPNetConfig, kw: dict[str, object]) -> CSPNetForImageClassification:
    return CSPNetForImageClassification(
        CSPNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# Backbones (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base", family="cspnet", model_type="cspnet",
    model_class=CSPNet, default_config=_CFG_CSPRESNET_50, params=21_620_000,
)
def cspresnet_50(pretrained: bool = False, **overrides: object) -> CSPNet:
    r"""CSPResNet-50 backbone — ResNet-50 with CSP-wrapped stages."""
    return _b(_CFG_CSPRESNET_50, overrides)


@register_model(
    task="base", family="cspnet", model_type="cspnet",
    model_class=CSPNet, default_config=_CFG_CSPRESNEXT_50, params=20_570_000,
)
def cspresnext_50(pretrained: bool = False, **overrides: object) -> CSPNet:
    r"""CSPResNeXt-50 backbone — ResNeXt-50 (32×4d) with CSP wrap."""
    return _b(_CFG_CSPRESNEXT_50, overrides)


@register_model(
    task="base", family="cspnet", model_type="cspnet",
    model_class=CSPNet, default_config=_CFG_CSPDARKNET_53, params=27_610_000,
)
def cspdarknet_53(pretrained: bool = False, **overrides: object) -> CSPNet:
    r"""CSPDarknet-53 backbone — Darknet-53 (YOLOv3 base) with CSP wrap."""
    return _b(_CFG_CSPDARKNET_53, overrides)


# ---------------------------------------------------------------------------
# Classifiers (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(  # type: ignore[arg-type]
    task="image-classification", family="cspnet", model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_CSPRESNET_50, params=21_620_000,
)
def cspresnet_50_cls(
    pretrained: bool | str = False,
    *,
    weights: CSPResNet50Weights | None = None,
    **overrides: object,
) -> CSPNetForImageClassification:
    r"""CSPResNet-50 image classifier — paper Table 1, **21.6M** params,
    ImageNet-1k acc@1 ≈ 76.2% (timm ``ra_in1k`` recipe)."""
    entry = weights_mod.resolve_weights(CSPResNet50Weights, pretrained, weights)
    model = _c(_CFG_CSPRESNET_50, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cspresnet_50_cls")
    return model


@register_model(  # type: ignore[arg-type]
    task="image-classification", family="cspnet", model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_CSPRESNEXT_50, params=20_570_000,
)
def cspresnext_50_cls(
    pretrained: bool | str = False,
    *,
    weights: CSPResNeXt50Weights | None = None,
    **overrides: object,
) -> CSPNetForImageClassification:
    r"""CSPResNeXt-50 image classifier — paper Table 1, **20.6M**
    params, ImageNet-1k acc@1 ≈ 80.0% (timm ``ra_in1k`` recipe)."""
    entry = weights_mod.resolve_weights(CSPResNeXt50Weights, pretrained, weights)
    model = _c(_CFG_CSPRESNEXT_50, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cspresnext_50_cls")
    return model


@register_model(  # type: ignore[arg-type]
    task="image-classification", family="cspnet", model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_CSPDARKNET_53, params=27_610_000,
)
def cspdarknet_53_cls(
    pretrained: bool | str = False,
    *,
    weights: CSPDarknet53Weights | None = None,
    **overrides: object,
) -> CSPNetForImageClassification:
    r"""CSPDarknet-53 image classifier — paper Table 1, **27.6M**
    params, ImageNet-1k acc@1 ≈ 80.1% (timm ``ra_in1k`` recipe)."""
    entry = weights_mod.resolve_weights(CSPDarknet53Weights, pretrained, weights)
    model = _c(_CFG_CSPDARKNET_53, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="cspdarknet_53_cls")
    return model
