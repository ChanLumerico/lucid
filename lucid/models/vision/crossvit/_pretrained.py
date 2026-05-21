"""Registry factories for CrossViT variants."""

from lucid.models._registry import register_model
from lucid.models.vision.crossvit._config import CrossViTConfig
from lucid.models.vision.crossvit._model import CrossViT, CrossViTForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# Paper: Chen et al., 2021 — "CrossViT: Cross-Attention Multi-Scale Vision
# Transformer for Image Classification"
#
# Two-branch ViT: small patch (P_S) + large patch (P_L).
# small_heads = small_dim // 64, large_heads = large_dim // 64 (typical).
# depth = number of CrossViT stages (each stage = self-attn + cross-attn).
# ---------------------------------------------------------------------------

# CrossViT-9  (~8.6 M) — small variant from paper
_CFG_9 = CrossViTConfig(
    depth=3,
    small_dim=128,
    large_dim=256,
    small_patch=12,
    large_patch=16,
    small_heads=4,
    large_heads=4,
    mlp_ratio=3.0,
)

# CrossViT-Tiny (~7.0 M)
_CFG_TINY = CrossViTConfig(
    depth=1,
    small_dim=96,
    large_dim=192,
    small_patch=12,
    large_patch=16,
    small_heads=3,
    large_heads=3,
    mlp_ratio=4.0,
)

# CrossViT-Small (~26.9 M)
_CFG_SMALL = CrossViTConfig(
    depth=4,
    small_dim=192,
    large_dim=384,
    small_patch=12,
    large_patch=16,
    small_heads=6,
    large_heads=6,
    mlp_ratio=4.0,
)

# CrossViT-Base (~105 M) — deeper / wider variant
_CFG_BASE = CrossViTConfig(
    depth=10,
    small_dim=192,
    large_dim=384,
    small_patch=12,
    large_patch=16,
    small_heads=6,
    large_heads=6,
    mlp_ratio=4.0,
)

# CrossViT-15 (~27.5 M)
_CFG_15 = CrossViTConfig(
    depth=3,
    small_dim=192,
    large_dim=384,
    small_patch=12,
    large_patch=16,
    small_heads=6,
    large_heads=6,
    mlp_ratio=3.0,
)

# CrossViT-18 (~43.3 M)
_CFG_18 = CrossViTConfig(
    depth=3,
    small_dim=224,
    large_dim=448,
    small_patch=12,
    large_patch=16,
    small_heads=7,
    large_heads=7,
    mlp_ratio=3.0,
)


def _b(cfg: CrossViTConfig, kw: dict[str, object]) -> CrossViT:
    return CrossViT(CrossViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: CrossViTConfig, kw: dict[str, object]) -> CrossViTForImageClassification:
    return CrossViTForImageClassification(
        CrossViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_9,
)
def crossvit_9(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-9 backbone (Chen et al., 2021).

    Builds the canonical *CrossViT-9* configuration: ``depth=3``,
    ``small_dim=128``, ``large_dim=256``, ``small_patch=12``,
    ``large_patch=16``, ``mlp_ratio=3.0``.  Approximately **8.6M
    parameters** — the smallest variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available in the model zoo.  Defaults to ``False``.
    **overrides : object
        Keyword overrides applied on top of the canonical CrossViT-9
        config.  Each override must match a field of
        :class:`CrossViTConfig`.

    Returns
    -------
    CrossViT
        A :class:`CrossViT` backbone returning a flat :math:`(B, 256)`
        large-branch CLS feature.

    Notes
    -----
    CrossViT-9 reaches **73.9% top-1 on ImageNet-1k** (Chen et al.,
    2021, Table 5).  See
    `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_9
    >>> model = crossvit_9()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 256)
    """
    return _b(_CFG_9, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_TINY,
)
def crossvit_tiny(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-Tiny backbone (Chen et al., 2021).

    Tiny dual-branch ViT with a *single* CrossViT stage
    (``depth=1``), ``small_dim=96``, ``large_dim=192``,
    ``mlp_ratio=4.0``.  Approximately **7.0M parameters** — useful for
    rapid prototyping and mobile baselines.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-Tiny config.

    Returns
    -------
    CrossViT
        A :class:`CrossViT` backbone returning a flat :math:`(B, 192)`
        large-branch CLS feature.

    Notes
    -----
    See `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_tiny
    >>> model = crossvit_tiny()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 192)
    """
    return _b(_CFG_TINY, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_SMALL,
)
def crossvit_small(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-Small backbone (Chen et al., 2021).

    Wider / deeper CrossViT with ``depth=4``, ``small_dim=192``,
    ``large_dim=384``, ``mlp_ratio=4.0``.  Approximately **26.9M
    parameters** — the standard small-class variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-Small config.

    Returns
    -------
    CrossViT
        A :class:`CrossViT` backbone returning a flat :math:`(B, 384)`
        large-branch CLS feature.

    Notes
    -----
    See `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_small
    >>> model = crossvit_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 384)
    """
    return _b(_CFG_SMALL, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_BASE,
)
def crossvit_base(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-Base backbone (Chen et al., 2021).

    Deeper variant with ``depth=10`` CrossViT stages,
    ``small_dim=192``, ``large_dim=384``, ``mlp_ratio=4.0``.
    Approximately **105M parameters** — trades parameter count for
    accuracy by lengthening the trunk rather than widening it.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-Base config.

    Returns
    -------
    CrossViT
        A :class:`CrossViT` backbone returning a flat :math:`(B, 384)`
        large-branch CLS feature.

    Notes
    -----
    See `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_base
    >>> model = crossvit_base()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 384)
    """
    return _b(_CFG_BASE, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_15,
)
def crossvit_15(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-15 backbone (Chen et al., 2021).

    Builds the *CrossViT-15* configuration: ``depth=3``,
    ``small_dim=192``, ``large_dim=384``, ``small_heads=6``,
    ``large_heads=6``, ``mlp_ratio=3.0``.  Approximately **27.5M
    parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-15 config.

    Returns
    -------
    CrossViT
        A :class:`CrossViT` backbone returning a flat :math:`(B, 384)`
        large-branch CLS feature.

    Notes
    -----
    CrossViT-15 reaches **81.5% top-1 on ImageNet-1k** (Chen et al.,
    2021, Table 5).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_15
    >>> model = crossvit_15()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 384)
    """
    return _b(_CFG_15, overrides)


@register_model(
    task="base",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViT,
    default_config=_CFG_18,
)
def crossvit_18(pretrained: bool = False, **overrides: object) -> CrossViT:
    r"""CrossViT-18 backbone (Chen et al., 2021).

    Builds the *CrossViT-18* configuration: ``depth=3``,
    ``small_dim=224``, ``large_dim=448``, ``small_heads=7``,
    ``large_heads=7``, ``mlp_ratio=3.0``.  Approximately **43.3M
    parameters** — the largest variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-18 config.

    Returns
    -------
    CrossViT
        A :class:`CrossViT` backbone returning a flat :math:`(B, 448)`
        large-branch CLS feature.

    Notes
    -----
    CrossViT-18 reaches **82.5% top-1 on ImageNet-1k** (Chen et al.,
    2021, Table 5).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_18
    >>> model = crossvit_18()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 448)
    """
    return _b(_CFG_18, overrides)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_9,
)
def crossvit_9_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-9 image classifier (Chen et al., 2021).

    Combines the :func:`crossvit_9` backbone with two per-branch
    linear heads whose logits are averaged (paper §3.3).  Default
    output is ``num_classes=1000`` (ImageNet-1k).  ~8.6M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-9 config.

    Returns
    -------
    CrossViTForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CrossViT-9 reaches **73.9% top-1 on ImageNet-1k** (Chen et al.,
    2021, Table 5).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_9_cls
    >>> model = crossvit_9_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_9, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_TINY,
)
def crossvit_tiny_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-Tiny image classifier (Chen et al., 2021).

    Combines the :func:`crossvit_tiny` backbone (``depth=1``,
    ``small_dim=96``, ``large_dim=192``) with two per-branch linear
    heads whose logits are averaged.  ~7.0M parameters — useful as a
    fast baseline for mobile / latency-sensitive applications.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-Tiny config.

    Returns
    -------
    CrossViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    See `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_tiny_cls
    >>> model = crossvit_tiny_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_TINY, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_SMALL,
)
def crossvit_small_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-Small image classifier (Chen et al., 2021).

    Combines the :func:`crossvit_small` backbone (``depth=4``,
    ``small_dim=192``, ``large_dim=384``) with two per-branch linear
    heads whose logits are averaged.  ~26.9M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-Small config.

    Returns
    -------
    CrossViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    See `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_small_cls
    >>> model = crossvit_small_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_SMALL, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_BASE,
)
def crossvit_base_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-Base image classifier (Chen et al., 2021).

    Combines the :func:`crossvit_base` backbone (``depth=10``,
    ``small_dim=192``, ``large_dim=384``) with two per-branch linear
    heads whose logits are averaged.  ~105M parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-Base config.

    Returns
    -------
    CrossViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    See `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_base_cls
    >>> model = crossvit_base_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_BASE, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_15,
)
def crossvit_15_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-15 image classifier (Chen et al., 2021).

    Combines the :func:`crossvit_15` backbone (``depth=3``,
    ``small_dim=192``, ``large_dim=384``, ``mlp_ratio=3.0``) with two
    per-branch linear heads whose logits are averaged.  ~27.5M
    parameters.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-15 config.

    Returns
    -------
    CrossViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CrossViT-15 reaches **81.5% top-1 on ImageNet-1k** (Chen et al.,
    2021, Table 5).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_15_cls
    >>> model = crossvit_15_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_15, overrides)


@register_model(
    task="image-classification",
    family="crossvit",
    model_type="crossvit",
    model_class=CrossViTForImageClassification,
    default_config=_CFG_18,
)
def crossvit_18_cls(
    pretrained: bool = False, **overrides: object
) -> CrossViTForImageClassification:
    r"""CrossViT-18 image classifier (Chen et al., 2021).

    Combines the :func:`crossvit_18` backbone (``depth=3``,
    ``small_dim=224``, ``large_dim=448``, ``mlp_ratio=3.0``) with two
    per-branch linear heads whose logits are averaged.  ~43.3M
    parameters — the largest variant in the paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CrossViT-18 config.

    Returns
    -------
    CrossViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CrossViT-18 reaches **82.5% top-1 on ImageNet-1k** (Chen et al.,
    2021, Table 5) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.crossvit import crossvit_18_cls
    >>> model = crossvit_18_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_18, overrides)
