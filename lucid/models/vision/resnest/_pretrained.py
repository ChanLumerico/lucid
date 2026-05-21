"""Registry factories for ResNeSt."""

from lucid.models._registry import register_model
from lucid.models.vision.resnest._config import ResNeStConfig
from lucid.models.vision.resnest._model import (
    ResNeSt,
    ResNeStForImageClassification,
)

_CFG_14 = ResNeStConfig(layers=(1, 1, 1, 1), radix=2, stem_width=32)
_CFG_26 = ResNeStConfig(layers=(2, 2, 2, 2), radix=2, stem_width=32)
_CFG_50 = ResNeStConfig(layers=(3, 4, 6, 3), radix=2)
_CFG_101 = ResNeStConfig(layers=(3, 4, 23, 3), radix=2, stem_width=64)
_CFG_200 = ResNeStConfig(layers=(3, 24, 36, 3), radix=2, stem_width=64)
_CFG_269 = ResNeStConfig(layers=(3, 30, 48, 8), radix=2, stem_width=64)


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_14,
)
def resnest_14(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-14 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with block repeats ``[1, 1, 1, 1]``
    over four stages — a lightweight variant of the ResNeSt family
    targeted at low-latency deployments.  Uses ``radix = 2`` and
    ``stem_width = 32``.  Approximately 10.6M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-14 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).  Final-stage output is 2048
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_14
    >>> model = resnest_14()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = ResNeStConfig(**{**_CFG_14.__dict__, **overrides}) if overrides else _CFG_14
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_26,
)
def resnest_26(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-26 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with block repeats ``[2, 2, 2, 2]``
    over four stages — a middle-ground variant between the
    lightweight ResNeSt-14 and the canonical ResNeSt-50.  Uses
    ``radix = 2`` and ``stem_width = 32``.  Approximately 17.1M
    parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-26 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_26
    >>> model = resnest_26()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = ResNeStConfig(**{**_CFG_26.__dict__, **overrides}) if overrides else _CFG_26
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_50,
)
def resnest_50(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-50 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with ResNet-50 topology
    (``[3, 4, 6, 3]``), ``radix = 2``, and the canonical
    ``stem_width = 32`` deep stem.  Approximately 27.5M parameters
    and 81.1% ImageNet-1k top-1 accuracy in Zhang et al., 2022
    (Table 4) — roughly 5 points higher than plain ResNet-50 at
    the same parameter budget.  The default ResNeSt configuration.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`
        (e.g. ``radix=4`` to widen the split-attention).

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_50
    >>> model = resnest_50()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = ResNeStConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_101,
)
def resnest_101(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-101 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with ResNet-101 topology
    (``[3, 4, 23, 3]``), ``radix = 2``, and a wider
    ``stem_width = 64`` deep stem.  Approximately 48.3M parameters
    and 82.8% ImageNet-1k top-1 accuracy in Zhang et al., 2022
    (Table 4).  Higher-accuracy drop-in replacement for ResNeSt-50.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-101 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_101
    >>> model = resnest_101()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = ResNeStConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_200,
)
def resnest_200(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-200 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with the deep topology
    ``[3, 24, 36, 3]``, ``radix = 2``, and ``stem_width = 64``.
    Approximately 70.2M parameters and 83.9% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-200 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_200
    >>> model = resnest_200()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = ResNeStConfig(**{**_CFG_200.__dict__, **overrides}) if overrides else _CFG_200
    return ResNeSt(cfg)


@register_model(
    task="base",
    family="resnest",
    model_type="resnest",
    model_class=ResNeSt,
    default_config=_CFG_269,
)
def resnest_269(pretrained: bool = False, **overrides: object) -> ResNeSt:
    r"""ResNeSt-269 feature-extracting backbone (no classification head).

    Builds a :class:`ResNeSt` with the very deep topology
    ``[3, 30, 48, 8]``, ``radix = 2``, and ``stem_width = 64`` —
    the deepest variant from Zhang et al., 2022.  Approximately
    110.9M parameters and 84.5% ImageNet-1k top-1 accuracy
    (Table 4) at 416×416 input resolution.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeSt
        Backbone with the ResNeSt-269 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.  Memory footprint
    is substantial; consider ``zero_init_residual=True`` for
    stable large-batch training.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_269
    >>> model = resnest_269()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = ResNeStConfig(**{**_CFG_269.__dict__, **overrides}) if overrides else _CFG_269
    return ResNeSt(cfg)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_14,
)
def resnest_14_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    r"""ResNeSt-14 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-14 backbone (block repeats ``[1, 1, 1, 1]``) followed
    by global average pooling and a linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    Approximately 10.6M parameters — the lightweight variant of
    the ResNeSt family.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-14 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_14_cls
    >>> model = resnest_14_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    cfg = ResNeStConfig(**{**_CFG_14.__dict__, **overrides}) if overrides else _CFG_14
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_26,
)
def resnest_26_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    r"""ResNeSt-26 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-26 backbone (block repeats ``[2, 2, 2, 2]``) followed
    by global average pooling and a linear projection.
    Approximately 17.1M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-26 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_26_cls
    >>> model = resnest_26_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = ResNeStConfig(**{**_CFG_26.__dict__, **overrides}) if overrides else _CFG_26
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_50,
)
def resnest_50_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    r"""ResNeSt-50 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    canonical ResNeSt-50 backbone (``[3, 4, 6, 3]``, ``radix = 2``)
    followed by global average pooling and a linear projection.
    Approximately 27.5M parameters and 81.1% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4) — the canonical
    ResNeSt configuration.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_50_cls
    >>> model = resnest_50_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    cfg = ResNeStConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_101,
)
def resnest_101_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    r"""ResNeSt-101 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-101 backbone (``[3, 4, 23, 3]``, ``stem_width = 64``)
    followed by global average pooling and a linear classifier.
    Approximately 48.3M parameters and 82.8% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-101 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_101_cls
    >>> model = resnest_101_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = ResNeStConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_200,
)
def resnest_200_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    r"""ResNeSt-200 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    ResNeSt-200 backbone (``[3, 24, 36, 3]``, ``stem_width = 64``)
    followed by global average pooling and a linear classifier.
    Approximately 70.2M parameters and 83.9% ImageNet-1k top-1
    accuracy in Zhang et al., 2022 (Table 4).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-200 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_200_cls
    >>> model = resnest_200_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = ResNeStConfig(**{**_CFG_200.__dict__, **overrides}) if overrides else _CFG_200
    return ResNeStForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="resnest",
    model_type="resnest",
    model_class=ResNeStForImageClassification,
    default_config=_CFG_269,
)
def resnest_269_cls(
    pretrained: bool = False, **overrides: object
) -> ResNeStForImageClassification:
    r"""ResNeSt-269 image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeStForImageClassification` with the
    deepest paper-cited backbone (``[3, 30, 48, 8]``,
    ``stem_width = 64``) followed by global average pooling and a
    linear classifier.  Approximately 110.9M parameters and 84.5%
    ImageNet-1k top-1 accuracy in Zhang et al., 2022 (Table 4) at
    416×416 input resolution.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeStConfig`.

    Returns
    -------
    ResNeStForImageClassification
        Classifier with the ResNeSt-269 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Zhang et al., "ResNeSt: Split-Attention Networks", CVPR
    Workshops 2022 (arXiv:2004.08955), Table 4.  Memory footprint
    is substantial.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnest import resnest_269_cls
    >>> model = resnest_269_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = ResNeStConfig(**{**_CFG_269.__dict__, **overrides}) if overrides else _CFG_269
    return ResNeStForImageClassification(cfg)
