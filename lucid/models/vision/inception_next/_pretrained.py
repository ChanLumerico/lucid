"""Registry factories for InceptionNeXt variants."""

from lucid.models._registry import register_model
from lucid.models.vision.inception_next._config import InceptionNeXtConfig
from lucid.models.vision.inception_next._model import (
    InceptionNeXt,
    InceptionNeXtForImageClassification,
)

_CFG_T = InceptionNeXtConfig(
    depths=(3, 3, 9, 3),
    dims=(96, 192, 384, 768),
    band_kernel=11,
)


def _b(cfg: InceptionNeXtConfig, kw: dict[str, object]) -> InceptionNeXt:
    return InceptionNeXt(InceptionNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(
    cfg: InceptionNeXtConfig, kw: dict[str, object]
) -> InceptionNeXtForImageClassification:
    return InceptionNeXtForImageClassification(
        InceptionNeXtConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXt,
    default_config=_CFG_T,
)
def inception_next_tiny(pretrained: bool = False, **overrides: object) -> InceptionNeXt:
    r"""InceptionNeXt-Tiny backbone (Yu et al., 2024).

    Builds the canonical *InceptionNeXt-T* configuration:
    ``depths=(3, 3, 9, 3)``, ``dims=(96, 192, 384, 768)``,
    ``band_kernel=11``.  Approximately **28M parameters** —
    ConvNeXt-T compatible, but with each :math:`7 \times 7` depthwise
    conv replaced by the four-branch InceptionDWConv mixer.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available in the model zoo.  Defaults to ``False``.
    **overrides : object
        Keyword overrides applied on top of the canonical
        InceptionNeXt-T config.  Each override must match a field of
        :class:`InceptionNeXtConfig`.

    Returns
    -------
    InceptionNeXt
        An :class:`InceptionNeXt` backbone returning a flat
        :math:`(B, 768)` feature.

    Notes
    -----
    InceptionNeXt-T matches ConvNeXt-T accuracy
    (**82.3% top-1 on ImageNet-1k**, Yu et al., 2024, Table 2) at
    noticeably lower latency.  See
    `arXiv:2303.16900 <https://arxiv.org/abs/2303.16900>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_tiny
    >>> model = inception_next_tiny()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768)
    """
    return _b(_CFG_T, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(
    task="image-classification",
    family="inception_next",
    model_type="inception_next",
    model_class=InceptionNeXtForImageClassification,
    default_config=_CFG_T,
)
def inception_next_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> InceptionNeXtForImageClassification:
    r"""InceptionNeXt-Tiny image classifier (Yu et al., 2024).

    Combines the :func:`inception_next_tiny` backbone with the
    reference MLP classifier head (``Linear → GELU → LayerNorm →
    Linear``).  Default output is ``num_classes=1000`` (ImageNet-1k).
    Approximately **28M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical InceptionNeXt-T config.

    Returns
    -------
    InceptionNeXtForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    InceptionNeXt-T reaches **82.3% top-1 on ImageNet-1k** (Yu et al.,
    2024, Table 2).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_next import inception_next_tiny_cls
    >>> model = inception_next_tiny_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_T, overrides)
