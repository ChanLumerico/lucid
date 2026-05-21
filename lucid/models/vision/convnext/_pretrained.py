"""Registry factories for ConvNeXt variants."""

from lucid.models._registry import register_model
from lucid.models.vision.convnext._config import ConvNeXtConfig
from lucid.models.vision.convnext._model import ConvNeXt, ConvNeXtForImageClassification

_CFG_T = ConvNeXtConfig(depths=(3, 3, 9, 3), dims=(96, 192, 384, 768))
_CFG_S = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(96, 192, 384, 768))
_CFG_B = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(128, 256, 512, 1024))
_CFG_L = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(192, 384, 768, 1536))
_CFG_XL = ConvNeXtConfig(depths=(3, 3, 27, 3), dims=(256, 512, 1024, 2048))


def _b(cfg: ConvNeXtConfig, kw: dict[str, object]) -> ConvNeXt:
    return ConvNeXt(ConvNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: ConvNeXtConfig, kw: dict[str, object]) -> ConvNeXtForImageClassification:
    return ConvNeXtForImageClassification(
        ConvNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_T,
)
def convnext_tiny(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Tiny backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-T* configuration: ``depths=(3, 3, 9, 3)``,
    ``dims=(96, 192, 384, 768)``.  The final stage produces a
    :math:`(B, 768, H/32, W/32)` feature map; ``forward_features``
    globally averages and LayerNorms to a :math:`(B, 768)` vector.
    Approximately **29M parameters** — Swin-T equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when available
        in the model zoo.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-T config.
        Each override must match a field of :class:`ConvNeXtConfig`.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    ConvNeXt-T reaches **82.1% top-1 on ImageNet-1k** (Liu et al., 2022,
    Table 1).  See `arXiv:2201.03545 <https://arxiv.org/abs/2201.03545>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_tiny
    >>> model = convnext_tiny()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768)
    """
    return _b(_CFG_T, overrides)


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_S,
)
def convnext_small(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Small backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-S* configuration: ``depths=(3, 3, 27, 3)``
    (3x deeper stage 3 than ConvNeXt-T), ``dims=(96, 192, 384, 768)``.
    Approximately **50M parameters** — Swin-S equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when available.
        Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-S config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 768)`
        feature.

    Notes
    -----
    ConvNeXt-S reaches **83.1% top-1 on ImageNet-1k** (Liu et al., 2022,
    Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_small
    >>> model = convnext_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 768)
    """
    return _b(_CFG_S, overrides)


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_B,
)
def convnext_base(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Base backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-B* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(128, 256, 512, 1024)``.  Final
    feature width is 1024.  Approximately **89M parameters** — Swin-B
    equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k or ImageNet-22k pretrained
        weights when available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-B config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 1024)`
        feature.

    Notes
    -----
    ConvNeXt-B reaches **83.8% top-1 on ImageNet-1k** (224x224) and
    **85.8% top-1** at 384x384 after ImageNet-22k pretraining (Liu
    et al., 2022, Tables 1 and 11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_base
    >>> model = convnext_base()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1024)
    """
    return _b(_CFG_B, overrides)


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_L,
)
def convnext_large(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-Large backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-L* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(192, 384, 768, 1536)``.  Final
    feature width is 1536.  Approximately **198M parameters** — Swin-L
    equivalent.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-L config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 1536)`
        feature.

    Notes
    -----
    ConvNeXt-L reaches **84.3% top-1 on ImageNet-1k** (224x224) and
    **86.6% top-1** at 384x384 after ImageNet-22k pretraining (Liu
    et al., 2022, Tables 1 and 11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_large
    >>> model = convnext_large()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1536)
    """
    return _b(_CFG_L, overrides)


@register_model(
    task="base",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXt,
    default_config=_CFG_XL,
)
def convnext_xlarge(pretrained: bool = False, **overrides: object) -> ConvNeXt:
    r"""ConvNeXt-XLarge backbone (Liu et al., 2022).

    Builds the canonical *ConvNeXt-XL* configuration:
    ``depths=(3, 3, 27, 3)``, ``dims=(256, 512, 1024, 2048)``.  Final
    feature width is 2048 — twice the ConvNeXt-B width.  Approximately
    **350M parameters** — the largest variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-XL config.

    Returns
    -------
    ConvNeXt
        A :class:`ConvNeXt` backbone returning a flat :math:`(B, 2048)`
        feature.

    Notes
    -----
    ConvNeXt-XL reaches **87.0% top-1 on ImageNet-1k** at 384x384
    fine-tune resolution after ImageNet-22k pretraining (Liu et al.,
    2022, Table 11) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_xlarge
    >>> model = convnext_xlarge()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 2048)
    """
    return _b(_CFG_XL, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_T,
)
def convnext_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Tiny image classifier (Liu et al., 2022).

    Combines the :func:`convnext_tiny` backbone with a global average
    pool + LayerNorm + single :class:`nn.Linear` classification head.
    Default output is ``num_classes=1000`` (ImageNet-1k).  ~29M
    parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-T config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-T reaches **82.1% top-1 on ImageNet-1k** (Liu et al.,
    2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_tiny_cls
    >>> model = convnext_tiny_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_T, overrides)


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_S,
)
def convnext_small_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Small image classifier (Liu et al., 2022).

    Combines the :func:`convnext_small` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(96, 192, 384, 768)``) with a global average pool +
    LayerNorm + linear classification head.  ~50M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-S config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-S reaches **83.1% top-1 on ImageNet-1k** (Liu et al.,
    2022, Table 1).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_small_cls
    >>> model = convnext_small_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_S, overrides)


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_B,
)
def convnext_base_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Base image classifier (Liu et al., 2022).

    Combines the :func:`convnext_base` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(128, 256, 512, 1024)``) with a global average pool +
    LayerNorm + linear classification head.  ~89M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k or ImageNet-22k pretrained
        weights when available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-B config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-B reaches **83.8% top-1 on ImageNet-1k** (224x224) and
    **85.8% top-1** at 384x384 after ImageNet-22k pretraining (Liu
    et al., 2022, Tables 1 and 11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_base_cls
    >>> model = convnext_base_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_B, overrides)


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_L,
)
def convnext_large_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-Large image classifier (Liu et al., 2022).

    Combines the :func:`convnext_large` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(192, 384, 768, 1536)``) with a global average pool +
    LayerNorm + linear classification head.  ~198M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-L config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-L reaches **84.3% top-1 on ImageNet-1k** (224x224) and
    **86.6% top-1** at 384x384 after ImageNet-22k pretraining (Liu
    et al., 2022, Tables 1 and 11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_large_cls
    >>> model = convnext_large_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_L, overrides)


@register_model(
    task="image-classification",
    family="convnext",
    model_type="convnext",
    model_class=ConvNeXtForImageClassification,
    default_config=_CFG_XL,
)
def convnext_xlarge_cls(
    pretrained: bool = False, **overrides: object
) -> ConvNeXtForImageClassification:
    r"""ConvNeXt-XLarge image classifier (Liu et al., 2022).

    Combines the :func:`convnext_xlarge` backbone (``depths=(3, 3, 27, 3)``,
    ``dims=(256, 512, 1024, 2048)``) with a global average pool +
    LayerNorm + linear classification head.  ~350M parameters — the
    largest ConvNeXt variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical ConvNeXt-XL config.

    Returns
    -------
    ConvNeXtForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    ConvNeXt-XL reaches **87.0% top-1 on ImageNet-1k** at 384x384
    fine-tune resolution after ImageNet-22k pretraining (Liu et al.,
    2022, Table 11) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.convnext import convnext_xlarge_cls
    >>> model = convnext_xlarge_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_XL, overrides)
