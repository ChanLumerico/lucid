"""Registry factories for all ResNeXt variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.resnext._config import ResNeXtConfig
from lucid.models.vision.resnext._model import ResNeXt, ResNeXtForImageClassification
from lucid.models.vision.resnext._weights import (
    ResNeXt50_32x4dWeights,
    ResNeXt101_32x4dWeights,
    ResNeXt101_32x8dWeights,
)

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_50_32x4d = ResNeXtConfig(layers=(3, 4, 6, 3), cardinality=32, width_per_group=4)
_CFG_101_32x4d = ResNeXtConfig(layers=(3, 4, 23, 3), cardinality=32, width_per_group=4)
_CFG_101_32x8d = ResNeXtConfig(layers=(3, 4, 23, 3), cardinality=32, width_per_group=8)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_50_32x4d,
)
def resnext_50_32x4d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    r"""ResNeXt-50 (32x4d) feature-extracting backbone.

    Builds a :class:`ResNeXt` with the paper-cited ResNeXt-50 (32x4d)
    topology: per-stage block counts ``(3, 4, 6, 3)`` (same as
    ResNet-50), cardinality :math:`C = 32`, width per group
    :math:`d = 4`.  Approximately 25.0 M parameters — within roughly
    1% of ResNet-50's parameter budget while achieving ≈1pp higher
    ImageNet top-1 accuracy (77.8% in Xie et al., 2017, Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig` to
        customise ``in_channels``, ``num_classes``, ``cardinality``,
        ``width_per_group``, or ``dropout``.

    Returns
    -------
    ResNeXt
        Backbone with the ResNeXt-50 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Xie et al., "Aggregated Residual Transformations for Deep
    Neural Networks", CVPR 2017, Table 1.  The ``32x4d`` shorthand
    encodes :math:`C \times d` — cardinality times width-per-group.
    Final-stage output is 2048 channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_50_32x4d
    >>> model = resnext_50_32x4d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 7, 7)
    (1, 2048, 7, 7)
    """
    cfg = (
        ResNeXtConfig(**{**_CFG_50_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_50_32x4d
    )
    return ResNeXt(cfg)


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_101_32x4d,
)
def resnext_101_32x4d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    r"""ResNeXt-101 (32x4d) feature-extracting backbone.

    Builds a :class:`ResNeXt` with the paper-cited ResNeXt-101 (32x4d)
    topology: per-stage block counts ``(3, 4, 23, 3)`` (same as
    ResNet-101), cardinality :math:`C = 32`, width per group
    :math:`d = 4`.  Approximately 44.2 M parameters; reaches 78.8%
    ImageNet top-1 in Xie et al., 2017 (Table 5).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.

    Returns
    -------
    ResNeXt
        Backbone with the ResNeXt-101 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Xie et al., CVPR 2017, Table 1.  Same depth as ResNet-101 with
    the :math:`3\times3` middle convolution split into 32 groups.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x4d
    >>> model = resnext_101_32x4d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 7, 7)
    (1, 2048, 7, 7)
    """
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x4d
    )
    return ResNeXt(cfg)


@register_model(
    task="base",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXt,
    default_config=_CFG_101_32x8d,
)
def resnext_101_32x8d(pretrained: bool = False, **overrides: object) -> ResNeXt:
    r"""ResNeXt-101 (32x8d) feature-extracting backbone.

    Builds a :class:`ResNeXt` with the higher-capacity ResNeXt-101
    (32x8d) topology: per-stage block counts ``(3, 4, 23, 3)``,
    cardinality :math:`C = 32`, width per group :math:`d = 8` (double
    the standard ResNeXt-101).  Approximately 88.8 M parameters — the
    widest of the canonical ResNeXt variants, widely used as the
    backbone for ImageNet-pretrained downstream models (e.g. Facebook's
    Instagram-pretrained ``ig_resnext_101_32x8d``).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.

    Returns
    -------
    ResNeXt
        Backbone with the ResNeXt-101 (32x8d) configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Xie et al., CVPR 2017.  This widened variant is included in
    the reference-framework model zoo because of its strong transfer
    learning performance.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x8d
    >>> model = resnext_101_32x8d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 7, 7)
    (1, 2048, 7, 7)
    """
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x8d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x8d
    )
    return ResNeXt(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(  # type: ignore[arg-type]  # reason: resnext_50_32x4d_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_50_32x4d,
)
def resnext_50_32x4d_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeXt50_32x4dWeights | None = None,
    **overrides: object,
) -> ResNeXtForImageClassification:
    r"""ResNeXt-50 (32x4d) image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeXtForImageClassification` with the
    paper-cited ResNeXt-50 (32x4d) topology and a
    :class:`~lucid.nn.Linear` classifier projecting 2048 →
    ``config.num_classes``.  Approximately 25.0 M parameters; the
    distributed ``IMAGENET1K_V2`` checkpoint reaches 81.198% ImageNet-1k
    top-1 with the improved training recipe.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`ResNeXt50_32x4dWeights.IMAGENET1K_V2`); a tag string
        (e.g. ``"IMAGENET1K_V2"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : ResNeXt50_32x4dWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``ResNeXt50_32x4dWeights.IMAGENET1K_V2``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.  Note:
        overriding ``num_classes`` away from the checkpoint's class
        count makes pretrained loading fail the strict key/shape check.

    Returns
    -------
    ResNeXtForImageClassification
        Classifier with the ResNeXt-50 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    Pretrained weights are converted from torchvision's
    ``ResNeXt50_32X4D_Weights.IMAGENET1K_V2`` and hosted on the Hugging
    Face Hub under ``lucid-dl/resnext-50-32x4d``.  The V2 preset uses a
    232 resize ahead of the 224 center crop.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_50_32x4d_cls
    >>> model = resnext_50_32x4d_cls(num_classes=100)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 100)

    Load ImageNet-pretrained weights:

    >>> model = resnext_50_32x4d_cls(pretrained=True)
    >>> from lucid.models.vision.resnext import ResNeXt50_32x4dWeights
    >>> model = resnext_50_32x4d_cls(weights=ResNeXt50_32x4dWeights.IMAGENET1K_V2)
    """
    entry = weights_mod.resolve_weights(ResNeXt50_32x4dWeights, pretrained, weights)
    cfg = (
        ResNeXtConfig(**{**_CFG_50_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_50_32x4d
    )
    model = ResNeXtForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnext_50_32x4d_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: resnext_101_32x4d_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_101_32x4d,
)
def resnext_101_32x4d_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeXt101_32x4dWeights | None = None,
    **overrides: object,
) -> ResNeXtForImageClassification:
    r"""ResNeXt-101 (32x4d) image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeXtForImageClassification` with the
    paper-cited ResNeXt-101 (32x4d) topology and a
    :class:`~lucid.nn.Linear` classifier projecting 2048 →
    ``config.num_classes``.  Approximately 44.2 M parameters; the
    distributed Gluon ``GLUON_IN1K`` checkpoint reaches 80.342%
    ImageNet-1k top-1.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`ResNeXt101_32x4dWeights.GLUON_IN1K`); a tag string
        (e.g. ``"GLUON_IN1K"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : ResNeXt101_32x4dWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``ResNeXt101_32x4dWeights.GLUON_IN1K``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.  Note:
        overriding ``num_classes`` away from the checkpoint's class
        count makes pretrained loading fail the strict key/shape check.

    Returns
    -------
    ResNeXtForImageClassification
        Classifier with the ResNeXt-101 (32x4d) configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    Pretrained weights are converted from the timm Gluon checkpoint
    ``resnext101_32x4d.gluon_in1k`` and hosted on the Hugging Face Hub
    under ``lucid-dl/resnext-101-32x4d``.  The Gluon preset uses bicubic
    interpolation with a 0.875 crop_pct (256 resize → 224 crop).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x4d_cls
    >>> model = resnext_101_32x4d_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = resnext_101_32x4d_cls(pretrained=True)
    >>> from lucid.models.vision.resnext import ResNeXt101_32x4dWeights
    >>> model = resnext_101_32x4d_cls(weights=ResNeXt101_32x4dWeights.GLUON_IN1K)
    """
    entry = weights_mod.resolve_weights(ResNeXt101_32x4dWeights, pretrained, weights)
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x4d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x4d
    )
    model = ResNeXtForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnext_101_32x4d_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: resnext_101_32x8d_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="resnext",
    model_type="resnext",
    model_class=ResNeXtForImageClassification,
    default_config=_CFG_101_32x8d,
)
def resnext_101_32x8d_cls(
    pretrained: bool | str = False,
    *,
    weights: ResNeXt101_32x8dWeights | None = None,
    **overrides: object,
) -> ResNeXtForImageClassification:
    r"""ResNeXt-101 (32x8d) image classifier (backbone + GAP + linear head).

    Builds a :class:`ResNeXtForImageClassification` with the
    high-capacity ResNeXt-101 (32x8d) topology and a
    :class:`~lucid.nn.Linear` classifier projecting 2048 →
    ``config.num_classes``.  Approximately 88.8 M parameters; the
    distributed ``IMAGENET1K_V2`` checkpoint reaches 82.834%
    ImageNet-1k top-1 with the improved training recipe.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag
        (:attr:`ResNeXt101_32x8dWeights.IMAGENET1K_V2`); a tag string
        (e.g. ``"IMAGENET1K_V2"``) → that specific checkpoint.  Mutually
        exclusive with ``weights`` (which wins if both are given).
    weights : ResNeXt101_32x8dWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``ResNeXt101_32x8dWeights.IMAGENET1K_V2``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`ResNeXtConfig`.  Note:
        overriding ``num_classes`` away from the checkpoint's class
        count makes pretrained loading fail the strict key/shape check.

    Returns
    -------
    ResNeXtForImageClassification
        Classifier with the ResNeXt-101 (32x8d) configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Xie et al., "Aggregated Residual Transformations for Deep
    Neural Networks", CVPR 2017.  Pretrained weights are converted from
    torchvision's ``ResNeXt101_32X8D_Weights.IMAGENET1K_V2`` and hosted
    on the Hugging Face Hub under ``lucid-dl/resnext-101-32x8d``.  The V2
    preset uses a 232 resize ahead of the 224 center crop.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.resnext import resnext_101_32x8d_cls
    >>> model = resnext_101_32x8d_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)

    Load ImageNet-pretrained weights:

    >>> model = resnext_101_32x8d_cls(pretrained=True)
    >>> from lucid.models.vision.resnext import ResNeXt101_32x8dWeights
    >>> model = resnext_101_32x8d_cls(weights=ResNeXt101_32x8dWeights.IMAGENET1K_V2)
    """
    entry = weights_mod.resolve_weights(ResNeXt101_32x8dWeights, pretrained, weights)
    cfg = (
        ResNeXtConfig(**{**_CFG_101_32x8d.__dict__, **overrides})
        if overrides
        else _CFG_101_32x8d
    )
    model = ResNeXtForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="resnext_101_32x8d_cls")
    return model
