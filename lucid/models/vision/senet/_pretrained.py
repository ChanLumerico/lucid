"""Registry factories for all SE-ResNet variants."""

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.senet._config import SENetConfig
from lucid.models.vision.senet._model import SENet, SENetForImageClassification
from lucid.models.vision.senet._weights import (
    SEResNet18Weights,
    SEResNet34Weights,
    SEResNet50Weights,
    SEResNet101Weights,
    SEResNet152Weights,
)

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_18 = SENetConfig(block_type="basic", layers=(2, 2, 2, 2), legacy_pool=True)
_CFG_34 = SENetConfig(block_type="basic", layers=(3, 4, 6, 3), legacy_pool=True)
_CFG_50 = SENetConfig(block_type="bottleneck", layers=(3, 4, 6, 3))
_CFG_101 = SENetConfig(block_type="bottleneck", layers=(3, 4, 23, 3), legacy_pool=True)
_CFG_152 = SENetConfig(block_type="bottleneck", layers=(3, 8, 36, 3), legacy_pool=True)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_18,
)
def se_resnet_18(pretrained: bool = False, **overrides: object) -> SENet:
    r"""SE-ResNet-18 feature-extracting backbone (no classification head).

    Builds an :class:`SENet` with ResNet-18 topology
    (:class:`_SEBasicBlock` blocks stacked ``[2, 2, 2, 2]``) plus
    a squeeze-and-excitation module at the end of every block.
    Approximately 11.8M parameters — about 1% more than plain
    ResNet-18 — at a typical 1.0–1.5 point ImageNet top-1
    accuracy improvement.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`
        (e.g. ``reduction=8`` to tighten the SE bottleneck).

    Returns
    -------
    SENet
        Backbone with the SE-ResNet-18 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_18
    >>> model = se_resnet_18()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    cfg = SENetConfig(**{**_CFG_18.__dict__, **overrides}) if overrides else _CFG_18
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_34,
)
def se_resnet_34(pretrained: bool = False, **overrides: object) -> SENet:
    r"""SE-ResNet-34 feature-extracting backbone (no classification head).

    Builds an :class:`SENet` with ResNet-34 topology
    (:class:`_SEBasicBlock` blocks stacked ``[3, 4, 6, 3]``) plus
    squeeze-and-excitation modules.  Approximately 21.9M
    parameters.  A drop-in upgrade over plain ResNet-34 with
    typical 1.0–1.5 point ImageNet top-1 gain.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENet
        Backbone with the SE-ResNet-34 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_34
    >>> model = se_resnet_34()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    cfg = SENetConfig(**{**_CFG_34.__dict__, **overrides}) if overrides else _CFG_34
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_50,
)
def se_resnet_50(pretrained: bool = False, **overrides: object) -> SENet:
    r"""SE-ResNet-50 feature-extracting backbone (no classification head).

    Builds an :class:`SENet` with ResNet-50 topology
    (:class:`_SEBottleneck` blocks stacked ``[3, 4, 6, 3]``) plus
    squeeze-and-excitation modules at the end of every bottleneck.
    Approximately 28.1M parameters — about 10% more than plain
    ResNet-50 — and 77.6% ImageNet-1k top-1 accuracy in Hu et al.,
    2018 (Table 2) versus 76.2% for plain ResNet-50.  The most
    widely deployed SE variant.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENet
        Backbone with the SE-ResNet-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.  Final-stage output is 2048
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_50
    >>> model = se_resnet_50()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = SENetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_101,
)
def se_resnet_101(pretrained: bool = False, **overrides: object) -> SENet:
    r"""SE-ResNet-101 feature-extracting backbone (no classification head).

    Builds an :class:`SENet` with ResNet-101 topology
    (:class:`_SEBottleneck` blocks stacked ``[3, 4, 23, 3]``) plus
    squeeze-and-excitation modules.  Approximately 49.3M
    parameters and 78.3% ImageNet-1k top-1 accuracy in Hu et al.,
    2018 (Table 2) versus 77.4% for plain ResNet-101.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENet
        Backbone with the SE-ResNet-101 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_101
    >>> model = se_resnet_101()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = SENetConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    return SENet(cfg)


@register_model(
    task="base",
    family="senet",
    model_type="senet",
    model_class=SENet,
    default_config=_CFG_152,
)
def se_resnet_152(pretrained: bool = False, **overrides: object) -> SENet:
    r"""SE-ResNet-152 feature-extracting backbone (no classification head).

    Builds an :class:`SENet` with ResNet-152 topology
    (:class:`_SEBottleneck` blocks stacked ``[3, 8, 36, 3]``) plus
    squeeze-and-excitation modules.  Approximately 66.8M
    parameters and 78.7% ImageNet-1k top-1 accuracy in Hu et al.,
    2018 (Table 2) — the deepest SE-ResNet variant in the original
    paper.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENet
        Backbone with the SE-ResNet-152 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_152
    >>> model = se_resnet_152()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = SENetConfig(**{**_CFG_152.__dict__, **overrides}) if overrides else _CFG_152
    return SENet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(  # type: ignore[arg-type]  # reason: se_resnet_18_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_18,
)
def se_resnet_18_cls(
    pretrained: bool | str = False,
    *,
    weights: SEResNet18Weights | None = None,
    **overrides: object,
) -> SENetForImageClassification:
    r"""SE-ResNet-18 image classifier (backbone + GAP + linear head).

    Builds an :class:`SENetForImageClassification` with the
    SE-ResNet-18 backbone (:class:`_SEBasicBlock` blocks stacked
    ``[2, 2, 2, 2]``) followed by global average pooling and a
    linear projection to ``config.num_classes`` (default 1000 for
    ImageNet-1k).  Approximately 11.8M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SEResNet18Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : SEResNet18Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENetForImageClassification
        Classifier with the SE-ResNet-18 configuration applied (or
        with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.  Pretrained weights are converted
    from timm's ``legacy_seresnet18.in1k`` and hosted on the Hugging
    Face Hub under ``lucid-dl/se-resnet-18``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_18_cls
    >>> model = se_resnet_18_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    entry = weights_mod.resolve_weights(SEResNet18Weights, pretrained, weights)
    cfg = SENetConfig(**{**_CFG_18.__dict__, **overrides}) if overrides else _CFG_18
    model = SENetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="se_resnet_18_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: se_resnet_34_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_34,
)
def se_resnet_34_cls(
    pretrained: bool | str = False,
    *,
    weights: SEResNet34Weights | None = None,
    **overrides: object,
) -> SENetForImageClassification:
    r"""SE-ResNet-34 image classifier (backbone + GAP + linear head).

    Builds an :class:`SENetForImageClassification` with the
    SE-ResNet-34 backbone (:class:`_SEBasicBlock` blocks stacked
    ``[3, 4, 6, 3]``) followed by global average pooling and a
    linear projection to ``config.num_classes``.  Approximately
    21.9M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SEResNet34Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : SEResNet34Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENetForImageClassification
        Classifier with the SE-ResNet-34 configuration applied (or
        with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.  Pretrained weights are converted
    from timm's ``legacy_seresnet34.in1k`` and hosted on the Hugging
    Face Hub under ``lucid-dl/se-resnet-34``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_34_cls
    >>> model = se_resnet_34_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SEResNet34Weights, pretrained, weights)
    cfg = SENetConfig(**{**_CFG_34.__dict__, **overrides}) if overrides else _CFG_34
    model = SENetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="se_resnet_34_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: se_resnet_50_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_50,
)
def se_resnet_50_cls(
    pretrained: bool | str = False,
    *,
    weights: SEResNet50Weights | None = None,
    **overrides: object,
) -> SENetForImageClassification:
    r"""SE-ResNet-50 image classifier (backbone + GAP + linear head).

    Builds an :class:`SENetForImageClassification` with the
    SE-ResNet-50 backbone (:class:`_SEBottleneck` blocks stacked
    ``[3, 4, 6, 3]``) followed by global average pooling and a
    linear projection to ``config.num_classes``.  Approximately
    28.1M parameters and 77.6% ImageNet-1k top-1 accuracy in
    Hu et al., 2018 (Table 2) — the canonical SE variant.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SEResNet50Weights.RA2_IN1K`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : SEResNet50Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    SENetForImageClassification
        Classifier with the SE-ResNet-50 configuration applied (or
        with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.  Pretrained weights are converted
    from timm's ``seresnet50.ra2_in1k`` (RandAugment recipe, 78.5
    top-1) and hosted on the Hugging Face Hub under
    ``lucid-dl/se-resnet-50``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_50_cls
    >>> model = se_resnet_50_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    entry = weights_mod.resolve_weights(SEResNet50Weights, pretrained, weights)
    cfg = SENetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    model = SENetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="se_resnet_50_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: se_resnet_101_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_101,
)
def se_resnet_101_cls(
    pretrained: bool | str = False,
    *,
    weights: SEResNet101Weights | None = None,
    **overrides: object,
) -> SENetForImageClassification:
    r"""SE-ResNet-101 image classifier (backbone + GAP + linear head).

    Builds an :class:`SENetForImageClassification` with the
    SE-ResNet-101 backbone (:class:`_SEBottleneck` blocks stacked
    ``[3, 4, 23, 3]``) followed by global average pooling and a
    linear classifier.  Approximately 49.3M parameters and 78.3%
    ImageNet-1k top-1 accuracy in Hu et al., 2018 (Table 2).

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SEResNet101Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : SEResNet101Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENetForImageClassification
        Classifier with the SE-ResNet-101 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.  Pretrained weights are converted
    from timm's ``legacy_seresnet101.in1k`` and hosted on the Hugging
    Face Hub under ``lucid-dl/se-resnet-101``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_101_cls
    >>> model = se_resnet_101_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SEResNet101Weights, pretrained, weights)
    cfg = SENetConfig(**{**_CFG_101.__dict__, **overrides}) if overrides else _CFG_101
    model = SENetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="se_resnet_101_cls")
    return model


@register_model(  # type: ignore[arg-type]  # reason: se_resnet_152_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="senet",
    model_type="senet",
    model_class=SENetForImageClassification,
    default_config=_CFG_152,
)
def se_resnet_152_cls(
    pretrained: bool | str = False,
    *,
    weights: SEResNet152Weights | None = None,
    **overrides: object,
) -> SENetForImageClassification:
    r"""SE-ResNet-152 image classifier (backbone + GAP + linear head).

    Builds an :class:`SENetForImageClassification` with the
    SE-ResNet-152 backbone (:class:`_SEBottleneck` blocks stacked
    ``[3, 8, 36, 3]``) followed by global average pooling and a
    linear classifier.  Approximately 66.8M parameters and 78.7%
    ImageNet-1k top-1 accuracy in Hu et al., 2018 (Table 2) — the
    deepest SE-ResNet variant in the original paper.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SEResNet152Weights.IN1K`); a tag
        string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : SEResNet152Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SENetConfig`.

    Returns
    -------
    SENetForImageClassification
        Classifier with the SE-ResNet-152 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Hu et al., "Squeeze-and-Excitation Networks", CVPR 2018
    (arXiv:1709.01507), Table 2.  Pretrained weights are converted
    from timm's ``legacy_seresnet152.in1k`` and hosted on the Hugging
    Face Hub under ``lucid-dl/se-resnet-152``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.senet import se_resnet_152_cls
    >>> model = se_resnet_152_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SEResNet152Weights, pretrained, weights)
    cfg = SENetConfig(**{**_CFG_152.__dict__, **overrides}) if overrides else _CFG_152
    model = SENetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="se_resnet_152_cls")
    return model
