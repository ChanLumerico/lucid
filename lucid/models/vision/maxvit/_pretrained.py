"""Registry factories for MaxViT variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.maxvit._config import MaxViTConfig
from lucid.models.vision.maxvit._model import MaxViT, MaxViTForImageClassification
from lucid.models.vision.maxvit._weights import (
    MaxViTBaseWeights,
    MaxViTLargeWeights,
    MaxViTSmallWeights,
    MaxViTTinyWeights,
)

_CFG_T = MaxViTConfig(
    depths=(2, 2, 5, 2),
    dims=(64, 128, 256, 512),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)

_CFG_S = MaxViTConfig(
    depths=(2, 2, 5, 2),
    dims=(96, 192, 384, 768),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
    stem_width=64,
)

_CFG_B = MaxViTConfig(
    depths=(2, 6, 14, 2),
    dims=(96, 192, 384, 768),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
    stem_width=64,
)

_CFG_L = MaxViTConfig(
    depths=(2, 6, 14, 2),
    dims=(128, 256, 512, 1024),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)

_CFG_XL = MaxViTConfig(
    depths=(2, 6, 14, 2),
    dims=(192, 384, 768, 1536),
    window_size=7,
    num_heads=32,
    mlp_ratio=4.0,
)


def _b(cfg: MaxViTConfig, kw: dict[str, object]) -> MaxViT:
    return MaxViT(MaxViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg)


def _c(cfg: MaxViTConfig, kw: dict[str, object]) -> MaxViTForImageClassification:
    return MaxViTForImageClassification(
        MaxViTConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ── Backbones ─────────────────────────────────────────────────────────────────


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_T,
)
def maxvit_tiny(pretrained: bool = False, **overrides: object) -> MaxViT:
    r"""MaxViT-Tiny backbone (Tu et al., 2022).

    Builds the canonical *MaxViT-Tiny* configuration:
    ``depths=(2, 2, 5, 2)``, ``dims=(64, 128, 256, 512)``,
    ``window_size=7``.  Approximately **31M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available in the model zoo.  Defaults to ``False``.
    **overrides : object
        Keyword overrides applied on top of the canonical MaxViT-Tiny
        config.  Each override must match a field of :class:`MaxViTConfig`.

    Returns
    -------
    MaxViT
        A :class:`MaxViT` backbone returning a flat :math:`(B, 512)`
        feature.

    Notes
    -----
    MaxViT-Tiny reaches **83.6% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).  See
    `arXiv:2204.01697 <https://arxiv.org/abs/2204.01697>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_tiny
    >>> model = maxvit_tiny()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 512)
    """
    return _b(_CFG_T, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_S,
)
def maxvit_small(pretrained: bool = False, **overrides: object) -> MaxViT:
    r"""MaxViT-Small backbone (Tu et al., 2022).

    Builds the canonical *MaxViT-Small* configuration: same depths
    as MaxViT-Tiny (``depths=(2, 2, 5, 2)``) but wider —
    ``dims=(96, 192, 384, 768)``.  Approximately **55.8M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-Small config.

    Returns
    -------
    MaxViT
        A :class:`MaxViT` backbone returning a flat :math:`(B, 768)`
        feature.

    Notes
    -----
    MaxViT-Small reaches **84.5% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_small
    >>> model = maxvit_small()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 768)
    """
    return _b(_CFG_S, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_B,
)
def maxvit_base(pretrained: bool = False, **overrides: object) -> MaxViT:
    r"""MaxViT-Base backbone (Tu et al., 2022).

    Builds the canonical *MaxViT-Base* configuration:
    ``depths=(2, 6, 14, 2)``, ``dims=(96, 192, 384, 768)``.
    Approximately **96.6M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k or ImageNet-22k pretrained
        weights when available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-Base config.

    Returns
    -------
    MaxViT
        A :class:`MaxViT` backbone returning a flat :math:`(B, 768)`
        feature.

    Notes
    -----
    MaxViT-Base reaches **84.9% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_base
    >>> model = maxvit_base()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 768)
    """
    return _b(_CFG_B, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_L,
)
def maxvit_large(pretrained: bool = False, **overrides: object) -> MaxViT:
    r"""MaxViT-Large backbone (Tu et al., 2022).

    Builds the canonical *MaxViT-Large* configuration:
    ``depths=(2, 6, 14, 2)``, ``dims=(128, 256, 512, 1024)``.
    Approximately **171.2M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-Large config.

    Returns
    -------
    MaxViT
        A :class:`MaxViT` backbone returning a flat :math:`(B, 1024)`
        feature.

    Notes
    -----
    MaxViT-Large reaches **85.2% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_large
    >>> model = maxvit_large()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1024)
    """
    return _b(_CFG_L, overrides)


@register_model(
    task="base",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViT,
    default_config=_CFG_XL,
)
def maxvit_xlarge(pretrained: bool = False, **overrides: object) -> MaxViT:
    r"""MaxViT-XLarge backbone (Tu et al., 2022).

    Builds the canonical *MaxViT-XLarge* configuration:
    ``depths=(2, 6, 14, 2)``, ``dims=(192, 384, 768, 1536)``.
    Approximately **383.7M parameters** — the largest variant in the
    paper.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-XLarge config.

    Returns
    -------
    MaxViT
        A :class:`MaxViT` backbone returning a flat :math:`(B, 1536)`
        feature.

    Notes
    -----
    MaxViT-XLarge reaches **85.5% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6) and **88.5%** when fine-tuned at 512x512
    after JFT-300M pretraining (Table 7) — the headline result of the
    paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_xlarge
    >>> model = maxvit_xlarge()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model.forward_features(x).shape
    (1, 1536)
    """
    return _b(_CFG_XL, overrides)


# ── Classifiers ───────────────────────────────────────────────────────────────


@register_model(  # type: ignore[arg-type]  # reason: maxvit_tiny_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_T,
)
def maxvit_tiny_cls(
    pretrained: bool | str = False,
    *,
    weights: MaxViTTinyWeights | None = None,
    **overrides: object,
) -> MaxViTForImageClassification:
    r"""MaxViT-Tiny image classifier (Tu et al., 2022).

    Combines the :func:`maxvit_tiny` backbone with the reference
    NormMLP classifier head (LayerNorm → ``Linear + Tanh`` → final
    Linear).  Default output is ``num_classes=1000`` (ImageNet-1k).
    ~31M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`MaxViTTinyWeights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MaxViTTinyWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-Tiny config.

    Returns
    -------
    MaxViTForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    MaxViT-Tiny reaches **83.6% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).  Pretrained weights are converted from
    timm's ``maxvit_tiny_tf_224.in1k`` and hosted under
    ``lucid-dl/maxvit-tiny``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_tiny_cls
    >>> model = maxvit_tiny_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(MaxViTTinyWeights, pretrained, weights)
    model = _c(_CFG_T, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="maxvit_tiny_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: maxvit_small_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_S,
)
def maxvit_small_cls(
    pretrained: bool | str = False,
    *,
    weights: MaxViTSmallWeights | None = None,
    **overrides: object,
) -> MaxViTForImageClassification:
    r"""MaxViT-Small image classifier (Tu et al., 2022).

    Combines the :func:`maxvit_small` backbone (``depths=(2, 2, 5, 2)``,
    ``dims=(96, 192, 384, 768)``) with the reference NormMLP head.
    ~68.9M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`MaxViTSmallWeights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MaxViTSmallWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-Small config.

    Returns
    -------
    MaxViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    MaxViT-Small reaches **84.5% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).  Pretrained weights are converted from
    timm's ``maxvit_small_tf_224.in1k`` and hosted under
    ``lucid-dl/maxvit-small``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_small_cls
    >>> model = maxvit_small_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(MaxViTSmallWeights, pretrained, weights)
    model = _c(_CFG_S, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="maxvit_small_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: maxvit_base_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_B,
)
def maxvit_base_cls(
    pretrained: bool | str = False,
    *,
    weights: MaxViTBaseWeights | None = None,
    **overrides: object,
) -> MaxViTForImageClassification:
    r"""MaxViT-Base image classifier (Tu et al., 2022).

    Combines the :func:`maxvit_base` backbone (``depths=(2, 6, 14, 2)``,
    ``dims=(96, 192, 384, 768)``) with the reference NormMLP head.
    ~119.5M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`MaxViTBaseWeights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MaxViTBaseWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-Base config.

    Returns
    -------
    MaxViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    MaxViT-Base reaches **84.9% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).  Pretrained weights are converted from
    timm's ``maxvit_base_tf_224.in1k`` and hosted under
    ``lucid-dl/maxvit-base``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_base_cls
    >>> model = maxvit_base_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(MaxViTBaseWeights, pretrained, weights)
    model = _c(_CFG_B, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="maxvit_base_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: maxvit_large_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_L,
)
def maxvit_large_cls(
    pretrained: bool | str = False,
    *,
    weights: MaxViTLargeWeights | None = None,
    **overrides: object,
) -> MaxViTForImageClassification:
    r"""MaxViT-Large image classifier (Tu et al., 2022).

    Combines the :func:`maxvit_large` backbone (``depths=(2, 6, 14, 2)``,
    ``dims=(128, 256, 512, 1024)``) with the reference NormMLP head.
    ~211.8M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`MaxViTLargeWeights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : MaxViTLargeWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-Large config.

    Returns
    -------
    MaxViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    MaxViT-Large reaches **85.2% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6).  Pretrained weights are converted from
    timm's ``maxvit_large_tf_224.in1k`` and hosted under
    ``lucid-dl/maxvit-large``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_large_cls
    >>> model = maxvit_large_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(MaxViTLargeWeights, pretrained, weights)
    model = _c(_CFG_L, overrides)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="maxvit_large_cls")
    return model


@register_model(
    task="image-classification",
    family="maxvit",
    model_type="maxvit",
    model_class=MaxViTForImageClassification,
    default_config=_CFG_XL,
)
def maxvit_xlarge_cls(
    pretrained: bool = False, **overrides: object
) -> MaxViTForImageClassification:
    r"""MaxViT-XLarge image classifier (Tu et al., 2022).

    Combines the :func:`maxvit_xlarge` backbone (``depths=(2, 6, 14, 2)``,
    ``dims=(192, 384, 768, 1536)``) with the reference NormMLP head.
    ~383.7M parameters — the largest MaxViT variant.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-22k or JFT-300M pretrained weights
        when available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical MaxViT-XLarge config.

    Returns
    -------
    MaxViTForImageClassification
        Classifier whose ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    MaxViT-XLarge reaches **85.5% top-1 on ImageNet-1k** at 224x224
    (Tu et al., 2022, Table 6) and **88.5%** at 512x512 after JFT-300M
    pretraining (Table 7) — the headline result of the paper.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maxvit import maxvit_xlarge_cls
    >>> model = maxvit_xlarge_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> model(x).logits.shape
    (1, 1000)
    """
    return _c(_CFG_XL, overrides)
