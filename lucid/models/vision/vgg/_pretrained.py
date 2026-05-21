"""Registry factories for VGG variants."""

from lucid.models._registry import register_model
from lucid.models.vision.vgg._config import VGGConfig
from lucid.models.vision.vgg._model import VGG, VGGForImageClassification

_CFG_11 = VGGConfig(arch=(1, 1, 2, 2, 2))
_CFG_13 = VGGConfig(arch=(2, 2, 2, 2, 2))
_CFG_16 = VGGConfig(arch=(2, 2, 3, 3, 3))
_CFG_19 = VGGConfig(arch=(2, 2, 4, 4, 4))
_CFG_11_BN = VGGConfig(arch=(1, 1, 2, 2, 2), batch_norm=True)
_CFG_13_BN = VGGConfig(arch=(2, 2, 2, 2, 2), batch_norm=True)
_CFG_16_BN = VGGConfig(arch=(2, 2, 3, 3, 3), batch_norm=True)
_CFG_19_BN = VGGConfig(arch=(2, 2, 4, 4, 4), batch_norm=True)


def _backbone(cfg: VGGConfig, overrides: dict[str, object]) -> VGG:
    if overrides:
        cfg = VGGConfig(**{**cfg.__dict__, **overrides})
    return VGG(cfg)


def _classifier(
    cfg: VGGConfig, overrides: dict[str, object]
) -> VGGForImageClassification:
    if overrides:
        cfg = VGGConfig(**{**cfg.__dict__, **overrides})
    return VGGForImageClassification(cfg)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_11
)
def vgg_11(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-11 feature-extracting backbone (config A, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-11 topology:
    per-block convolution counts ``(1, 1, 2, 2, 2)`` across the five
    blocks of widths ``(64, 128, 256, 512, 512)``.  Approximately
    132.9 M parameters in the full classifier variant (≈9.2 M in the
    convolutional trunk alone).  Reaches a top-5 ImageNet validation
    error of 9.4% in Simonyan & Zisserman, ICLR 2015 (Table 3,
    configuration A).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig` to
        customise individual fields (``in_channels``, ``num_classes``,
        ``dropout``).

    Returns
    -------
    VGG
        Backbone with the VGG-11 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, "Very Deep Convolutional Networks for
    Large-Scale Image Recognition", ICLR 2015, Table 1 (configuration A).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11
    >>> model = vgg_11()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_11, overrides)


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_13
)
def vgg_13(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-13 feature-extracting backbone (config B, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-13 topology:
    per-block convolution counts ``(2, 2, 2, 2, 2)``.  Approximately
    133.0 M parameters in the full classifier variant.  Reaches a top-5
    ImageNet validation error of 8.8% (Simonyan & Zisserman, ICLR 2015,
    Table 3, configuration B).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-13 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, ICLR 2015, Table 1 (configuration B).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13
    >>> model = vgg_13()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_13, overrides)


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_16
)
def vgg_16(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-16 feature-extracting backbone (config D, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-16 topology:
    per-block convolution counts ``(2, 2, 3, 3, 3)`` — adding a third
    :math:`3\times3` convolution to each of the deeper blocks 3/4/5.
    Approximately 138.4 M parameters in the full classifier variant
    (≈14.7 M in the convolutional trunk alone).  Reaches a top-5
    ImageNet validation error of 7.3% (Simonyan & Zisserman, ICLR 2015,
    Table 3, configuration D) — VGG-16 was the most widely deployed
    variant of the family.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-16 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, ICLR 2015, Table 1 (configuration D).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16
    >>> model = vgg_16()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_16, overrides)


@register_model(
    task="base", family="vgg", model_type="vgg", model_class=VGG, default_config=_CFG_19
)
def vgg_19(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-19 feature-extracting backbone (config E, no BatchNorm).

    Builds a :class:`VGG` with the paper-cited VGG-19 topology:
    per-block convolution counts ``(2, 2, 4, 4, 4)`` — four
    :math:`3\times3` convolutions in each of the deeper blocks.
    Approximately 143.7 M parameters in the full classifier variant.
    Reaches a top-5 ImageNet validation error of 7.5% (Simonyan &
    Zisserman, ICLR 2015, Table 3, configuration E) — slightly worse
    than VGG-16 due to optimisation difficulty at this depth without
    residual connections.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-19 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Simonyan & Zisserman, ICLR 2015, Table 1 (configuration E).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19
    >>> model = vgg_19()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 512, 7, 7)
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_19, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_11_BN,
)
def vgg_11_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-11 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-11 topology
    ``arch=(1, 1, 2, 2, 2)`` and a :class:`~lucid.nn.BatchNorm2d` layer
    after each Conv + ReLU pair.  BatchNorm was added in the
    timm / reference-framework reimplementations of VGG (not in the
    original paper) and substantially improves convergence speed and
    final accuracy; the topology is otherwise identical to
    :func:`vgg_11`.  Approximately 132.9 M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-11-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11_bn
    >>> model = vgg_11_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_11_BN, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_13_BN,
)
def vgg_13_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-13 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-13 topology
    ``arch=(2, 2, 2, 2, 2)`` and :class:`~lucid.nn.BatchNorm2d` after
    each Conv + ReLU pair.  Approximately 133.0 M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-13-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13_bn
    >>> model = vgg_13_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_13_BN, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_16_BN,
)
def vgg_16_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-16 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-16 topology
    ``arch=(2, 2, 3, 3, 3)`` and :class:`~lucid.nn.BatchNorm2d` after
    each Conv + ReLU pair.  Approximately 138.4 M parameters.  Often
    preferred over plain :func:`vgg_16` for downstream fine-tuning
    because BatchNorm stabilises gradient statistics in the early conv
    stack.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-16-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16_bn
    >>> model = vgg_16_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_16_BN, overrides)


@register_model(
    task="base",
    family="vgg",
    model_type="vgg",
    model_class=VGG,
    default_config=_CFG_19_BN,
)
def vgg_19_bn(pretrained: bool = False, **overrides: object) -> VGG:
    r"""VGG-19 with BatchNorm — feature-extracting backbone.

    Builds a :class:`VGG` with the VGG-19 topology
    ``arch=(2, 2, 4, 4, 4)`` and :class:`~lucid.nn.BatchNorm2d` after
    each Conv + ReLU pair.  Approximately 143.7 M parameters.  The
    deepest VGG variant; BatchNorm is particularly helpful at this
    depth.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGG
        Backbone with the VGG-19-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19_bn
    >>> model = vgg_19_bn()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    return _backbone(_CFG_19_BN, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_11,
)
def vgg_11_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-11 image classifier (config A, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-11 topology (``arch=(1, 1, 2, 2, 2)``) followed by the two
    4096-dim fully-connected layers and a final linear projection to
    ``config.num_classes``.  Approximately 132.9 M parameters total
    (≈124 M of which sit in the two large FC layers).  Reaches a top-5
    ImageNet validation error of 9.4% (Simonyan & Zisserman, ICLR 2015,
    Table 3, configuration A).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig` — use
        ``num_classes=N`` to retarget the head.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-11 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11_cls
    >>> model = vgg_11_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _classifier(_CFG_11, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_13,
)
def vgg_13_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-13 image classifier (config B, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-13 topology (``arch=(2, 2, 2, 2, 2)``) followed by the standard
    4096 → 4096 → ``num_classes`` FC head.  Approximately 133.0 M
    parameters total.  Reaches a top-5 ImageNet validation error of
    8.8% (Simonyan & Zisserman, ICLR 2015, Table 3, configuration B).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-13 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13_cls
    >>> model = vgg_13_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _classifier(_CFG_13, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_16,
)
def vgg_16_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-16 image classifier (config D, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-16 topology (``arch=(2, 2, 3, 3, 3)``) followed by the standard
    4096 → 4096 → ``num_classes`` FC head.  Approximately 138.4 M
    parameters total and a top-5 ImageNet validation error of 7.3%
    (Simonyan & Zisserman, ICLR 2015, Table 3, configuration D) — the
    most widely used VGG variant.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-16 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16_cls
    >>> model = vgg_16_cls(num_classes=100)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 100)
    """
    return _classifier(_CFG_16, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_19,
)
def vgg_19_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-19 image classifier (config E, no BatchNorm).

    Builds a :class:`VGGForImageClassification` with the paper-cited
    VGG-19 topology (``arch=(2, 2, 4, 4, 4)``) followed by the standard
    4096 → 4096 → ``num_classes`` FC head.  Approximately 143.7 M
    parameters total.  Reaches a top-5 ImageNet validation error of
    7.5% (Simonyan & Zisserman, ICLR 2015, Table 3, configuration E).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-19 configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19_cls
    >>> model = vgg_19_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _classifier(_CFG_19, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_11_BN,
)
def vgg_11_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-11 with BatchNorm — image classifier.

    Same topology as :func:`vgg_11_cls` but with
    :class:`~lucid.nn.BatchNorm2d` after each Conv + ReLU pair.
    BatchNorm was added in the timm / reference-framework
    reimplementations of VGG (not in the original paper) and
    substantially improves convergence speed and final accuracy.
    Approximately 132.9 M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-11-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_11_bn_cls
    >>> model = vgg_11_bn_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _classifier(_CFG_11_BN, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_13_BN,
)
def vgg_13_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-13 with BatchNorm — image classifier.

    Same topology as :func:`vgg_13_cls` but with
    :class:`~lucid.nn.BatchNorm2d` after each Conv + ReLU pair.
    Approximately 133.0 M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-13-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_13_bn_cls
    >>> model = vgg_13_bn_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _classifier(_CFG_13_BN, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_16_BN,
)
def vgg_16_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-16 with BatchNorm — image classifier.

    Same topology as :func:`vgg_16_cls` but with
    :class:`~lucid.nn.BatchNorm2d` after each Conv + ReLU pair.
    Approximately 138.4 M parameters.  Often preferred over plain
    :func:`vgg_16_cls` for downstream fine-tuning.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-16-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_16_bn_cls
    >>> model = vgg_16_bn_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _classifier(_CFG_16_BN, overrides)


@register_model(
    task="image-classification",
    family="vgg",
    model_type="vgg",
    model_class=VGGForImageClassification,
    default_config=_CFG_19_BN,
)
def vgg_19_bn_cls(
    pretrained: bool = False, **overrides: object
) -> VGGForImageClassification:
    r"""VGG-19 with BatchNorm — image classifier.

    Same topology as :func:`vgg_19_cls` but with
    :class:`~lucid.nn.BatchNorm2d` after each Conv + ReLU pair.
    Approximately 143.7 M parameters.  The deepest VGG variant —
    BatchNorm is especially valuable at this depth.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`VGGConfig`.

    Returns
    -------
    VGGForImageClassification
        Classifier with the VGG-19-BN configuration applied (or with
        ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.vgg import vgg_19_bn_cls
    >>> model = vgg_19_bn_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    return _classifier(_CFG_19_BN, overrides)
