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
    r"""DenseNet-121 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-121
    topology: per-block dense-layer counts ``(6, 12, 24, 16)``,
    growth rate :math:`k = 32`, initial conv channels 64.
    Approximately 8.0 M parameters total — by far the most
    parameter-efficient of the ImageNet DenseNets, reaching a top-1
    error of 25.0% (Huang et al., 2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`
        (``growth_rate``, ``dropout_rate``, ``in_channels``, etc.).

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-121 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., "Densely Connected Convolutional Networks",
    CVPR 2017, Table 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_121
    >>> model = densenet_121()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1024, 1, 1)
    (1, 1024, 1, 1)
    """
    return _b(_CFG_121, overrides)


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_169,
)
def densenet_169(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-169 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-169
    topology: per-block dense-layer counts ``(6, 12, 32, 32)``, growth
    rate :math:`k = 32`, initial conv channels 64.  Approximately
    14.3 M parameters total.  Reaches a top-1 ImageNet error of 23.8%
    (Huang et al., 2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-169 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., CVPR 2017, Table 1.  Final pre-classifier feature
    width is 1664 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_169
    >>> model = densenet_169()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1664, 1, 1)
    (1, 1664, 1, 1)
    """
    return _b(_CFG_169, overrides)


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_201,
)
def densenet_201(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-201 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-201
    topology: per-block dense-layer counts ``(6, 12, 48, 32)``, growth
    rate :math:`k = 32`, initial conv channels 64.  Approximately
    20.0 M parameters total.  Reaches a top-1 ImageNet error of 22.6%
    (Huang et al., 2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-201 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., CVPR 2017, Table 1.  Final pre-classifier feature
    width is 1920 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_201
    >>> model = densenet_201()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1920, 1, 1)
    (1, 1920, 1, 1)
    """
    return _b(_CFG_201, overrides)


@register_model(
    task="base",
    family="densenet",
    model_type="densenet",
    model_class=DenseNet,
    default_config=_CFG_264,
)
def densenet_264(pretrained: bool = False, **overrides: object) -> DenseNet:
    r"""DenseNet-264 feature-extracting backbone (no classification head).

    Builds a :class:`DenseNet` with the paper-cited DenseNet-264
    topology: per-block dense-layer counts ``(6, 12, 64, 48)``, growth
    rate :math:`k = 32`, initial conv channels 64.  Approximately
    33.3 M parameters total — the deepest DenseNet evaluated in the
    paper, reaching a top-1 ImageNet error of 22.2% (Huang et al.,
    2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNet
        Backbone with the DenseNet-264 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Huang et al., CVPR 2017, Table 1.  Final pre-classifier feature
    width is 2688 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_264
    >>> model = densenet_264()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2688, 1, 1)
    (1, 2688, 1, 1)
    """
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
    r"""DenseNet-121 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-121 topology and a single
    :class:`~lucid.nn.Linear` classifier projecting 1024 →
    ``config.num_classes``.  Approximately 8.0 M parameters; top-1
    ImageNet error 25.0% (Huang et al., 2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-121 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_121_cls
    >>> model = densenet_121_cls(num_classes=100)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 100)
    """
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
    r"""DenseNet-169 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-169 topology and a :class:`~lucid.nn.Linear`
    classifier projecting 1664 → ``config.num_classes``.
    Approximately 14.3 M parameters; top-1 ImageNet error 23.8%
    (Huang et al., 2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-169 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_169_cls
    >>> model = densenet_169_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
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
    r"""DenseNet-201 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-201 topology and a :class:`~lucid.nn.Linear`
    classifier projecting 1920 → ``config.num_classes``.
    Approximately 20.0 M parameters; top-1 ImageNet error 22.6%
    (Huang et al., 2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-201 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_201_cls
    >>> model = densenet_201_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
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
    r"""DenseNet-264 image classifier (backbone + GAP + linear head).

    Builds a :class:`DenseNetForImageClassification` with the
    paper-cited DenseNet-264 topology and a :class:`~lucid.nn.Linear`
    classifier projecting 2688 → ``config.num_classes``.
    Approximately 33.3 M parameters; the deepest DenseNet evaluated in
    the paper, with a top-1 ImageNet error of 22.2% (Huang et al.,
    2017, Table 2).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`DenseNetConfig`.

    Returns
    -------
    DenseNetForImageClassification
        Classifier with the DenseNet-264 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.densenet import densenet_264_cls
    >>> model = densenet_264_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _c(_CFG_264, overrides)
