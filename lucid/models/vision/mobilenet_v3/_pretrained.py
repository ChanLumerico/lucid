"""Registry factories for MobileNet v3."""

from lucid.models._registry import register_model
from lucid.models.vision.mobilenet_v3._config import MobileNetV3Config
from lucid.models.vision.mobilenet_v3._model import (
    MobileNetV3,
    MobileNetV3ForImageClassification,
)

_CFG_LARGE = MobileNetV3Config(variant="large")
_CFG_SMALL = MobileNetV3Config(variant="small")


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3,
    default_config=_CFG_LARGE,
)
def mobilenet_v3_large(pretrained: bool = False, **overrides: object) -> MobileNetV3:
    r"""MobileNet-v3-Large feature-extracting backbone.

    Builds a :class:`MobileNetV3` with the NAS-designed Large
    topology from Howard et al., 2019 (Table 1): 15 inverted-residual
    bottleneck blocks with selective SE attention and hard-swish in
    the deeper half, followed by a 1×1 expansion to 960 channels.
    Approximately 5.4M parameters and 75.2% ImageNet-1k top-1
    accuracy (Table 3) — the higher-accuracy variant of the family.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`
        (e.g. ``width_mult=0.75`` for a slimmer variant).

    Returns
    -------
    MobileNetV3
        Backbone with the MobileNet-v3-Large configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 1 (V3-Large spec) and Table 3
    (ImageNet accuracy).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_large
    >>> model = mobilenet_v3_large()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 960, 1, 1)
    """
    cfg = (
        MobileNetV3Config(**{**_CFG_LARGE.__dict__, **overrides})
        if overrides
        else _CFG_LARGE
    )
    return MobileNetV3(cfg)


@register_model(
    task="base",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3,
    default_config=_CFG_SMALL,
)
def mobilenet_v3_small(pretrained: bool = False, **overrides: object) -> MobileNetV3:
    r"""MobileNet-v3-Small feature-extracting backbone.

    Builds a :class:`MobileNetV3` with the NAS-designed Small
    topology from Howard et al., 2019 (Table 2): 11 inverted-residual
    bottleneck blocks with selective SE attention and aggressive
    hard-swish usage from the very first stage, followed by a 1×1
    expansion to 576 channels.  Approximately 2.9M parameters and
    67.4% ImageNet-1k top-1 accuracy (Table 3) — the lower-latency
    variant of the family, targeting tight mobile budgets.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`.

    Returns
    -------
    MobileNetV3
        Backbone with the MobileNet-v3-Small configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 2 (V3-Small spec) and Table 3
    (ImageNet accuracy).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_small
    >>> model = mobilenet_v3_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 576, 1, 1)
    """
    cfg = (
        MobileNetV3Config(**{**_CFG_SMALL.__dict__, **overrides})
        if overrides
        else _CFG_SMALL
    )
    return MobileNetV3(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3ForImageClassification,
    default_config=_CFG_LARGE,
)
def mobilenet_v3_large_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV3ForImageClassification:
    r"""MobileNet-v3-Large image classifier with the redesigned head.

    Builds a :class:`MobileNetV3ForImageClassification` with the
    Large topology (Howard et al., 2019) plus the redesigned
    inverted classification head (``GAP → 1×1 Conv → h-swish →
    Flatten → Dropout → Linear``).  Approximately 5.4M parameters
    and 75.2% ImageNet-1k top-1 accuracy (Table 3).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    MobileNetV3ForImageClassification
        Classifier with the MobileNet-v3-Large configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 1 and Table 3.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_large_cls
    >>> model = mobilenet_v3_large_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    cfg = (
        MobileNetV3Config(**{**_CFG_LARGE.__dict__, **overrides})
        if overrides
        else _CFG_LARGE
    )
    return MobileNetV3ForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="mobilenet_v3",
    model_type="mobilenet_v3",
    model_class=MobileNetV3ForImageClassification,
    default_config=_CFG_SMALL,
)
def mobilenet_v3_small_cls(
    pretrained: bool = False, **overrides: object
) -> MobileNetV3ForImageClassification:
    r"""MobileNet-v3-Small image classifier with the redesigned head.

    Builds a :class:`MobileNetV3ForImageClassification` with the
    Small topology (Howard et al., 2019) plus the redesigned
    inverted classification head.  Approximately 2.9M parameters
    and 67.4% ImageNet-1k top-1 accuracy (Table 3) — the
    latency-focused variant of the family.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`MobileNetV3Config`.

    Returns
    -------
    MobileNetV3ForImageClassification
        Classifier with the MobileNet-v3-Small configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Howard et al., "Searching for MobileNetV3", ICCV 2019
    (arXiv:1905.02244), Table 2 and Table 3.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mobilenet_v3 import mobilenet_v3_small_cls
    >>> model = mobilenet_v3_small_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = (
        MobileNetV3Config(**{**_CFG_SMALL.__dict__, **overrides})
        if overrides
        else _CFG_SMALL
    )
    return MobileNetV3ForImageClassification(cfg)
