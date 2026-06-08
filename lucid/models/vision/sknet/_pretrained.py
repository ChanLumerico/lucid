"""Registry factories for all SKNet variants."""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.sknet._config import SKNetConfig
from lucid.models.vision.sknet._model import SKNet, SKNetForImageClassification
from lucid.models.vision.sknet._weights import SKResNet18Weights, SKResNet34Weights

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

# sk_resnet_18 / sk_resnet_34:
#   basic block (expansion=1); the first 3×3 conv per block is a
#   SelectiveKernel unit (two parallel 3×3 branches + channel attention),
#   the second is a plain 3×3 conv.  Matches the reference ``skresnet18`` /
#   ``skresnet34`` recipe: ``split_input=True`` (each branch receives half
#   the input channels), ``rd_ratio=1/8`` with ``rd_divisor=16`` for the
#   attention bottleneck.  sk_resnet_18: ~11.96M params;
#   sk_resnet_34: ~22.28M params.
_CFG_SK18 = SKNetConfig(
    layers=(2, 2, 2, 2),
    block_type="basic",
    cardinality=1,
    base_width=64,
    split_input=True,
    rd_ratio=1.0 / 8,
    rd_divisor=16,
)
_CFG_SK34 = SKNetConfig(
    layers=(3, 4, 6, 3),
    block_type="basic",
    cardinality=1,
    base_width=64,
    split_input=True,
    rd_ratio=1.0 / 8,
    rd_divisor=16,
)

# sk_resnet_50 / sk_resnet_101:
#   cardinality=1, base_width=64, split_input=True  →  timm ``skresnet50``
#   25,803,160 parameters for sk_resnet_50_cls (1000-class head)
_CFG_SK50 = SKNetConfig(
    layers=(3, 4, 6, 3), cardinality=1, base_width=64, split_input=True
)
_CFG_SK101 = SKNetConfig(
    layers=(3, 4, 23, 3), cardinality=1, base_width=64, split_input=True
)

# sk_resnext_50_32x4d:
#   cardinality=32, base_width=4, split_input=False, rd_ratio=1/16, rd_divisor=32
#   Equivalent to the SKNet-50 entry in the original paper.  27,479,784 parameters.
_CFG_SK_RX50 = SKNetConfig(
    layers=(3, 4, 6, 3),
    cardinality=32,
    base_width=4,
    split_input=False,
    rd_ratio=1.0 / 16,
    rd_divisor=32,
)


# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK18,
)
def sk_resnet_18(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-18 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-18 topology
    (:class:`_SelectiveKernelBasic` blocks stacked ``[2, 2, 2, 2]``).
    Both :math:`3 \times 3` convolutions inside every block are
    replaced by Selective Kernel units, giving full SK treatment
    of the basic-block design.  Approximately 24.7M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-18 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).  Uses ``rd_ratio = 0.6`` for the attention
    bottleneck and ``split_input = False`` so each branch receives
    the full input — matching the SK-ResNet-18 budget reference.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_18
    >>> model = sk_resnet_18()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    cfg = (
        replace(_CFG_SK18, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK18
    )
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK34,
)
def sk_resnet_34(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-34 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-34 topology
    (:class:`_SelectiveKernelBasic` blocks stacked ``[3, 4, 6, 3]``).
    Both :math:`3 \times 3` convolutions inside every basic block
    are replaced by Selective Kernel units.  Approximately 46.9M
    parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-34 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_34
    >>> model = sk_resnet_34()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    cfg = (
        replace(_CFG_SK34, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK34
    )
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK50,
)
def sk_resnet_50(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-50 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-50 bottleneck topology
    (:class:`_SelectiveKernelBottleneck` blocks stacked
    ``[3, 4, 6, 3]``).  The central :math:`3 \times 3` of every
    bottleneck is replaced by a two-branch Selective Kernel unit
    with ``split_input=True`` (each branch receives half the
    channels — matching timm's ``skresnet50`` layout).
    Approximately 25.8M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1.  Final-stage output is 2048
    channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_50
    >>> model = sk_resnet_50()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = (
        replace(_CFG_SK50, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK50
    )
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK101,
)
def sk_resnet_101(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNet-101 feature-extracting backbone (no classification head).

    Builds an :class:`SKNet` with ResNet-101 bottleneck topology
    (:class:`_SelectiveKernelBottleneck` blocks stacked
    ``[3, 4, 23, 3]``).  Approximately 45M parameters.  Deeper
    variant of SK-ResNet-50 for higher-accuracy budgets.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNet-101 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_101
    >>> model = sk_resnet_101()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = (
        replace(_CFG_SK101, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK101
    )
    return SKNet(cfg)


@register_model(
    task="base",
    family="sknet",
    model_type="sknet",
    model_class=SKNet,
    default_config=_CFG_SK_RX50,
)
def sk_resnext_50_32x4d(pretrained: bool = False, **overrides: object) -> SKNet:
    r"""SK-ResNeXt-50 32×4d feature-extracting backbone (the paper's SKNet-50).

    Builds an :class:`SKNet` with ResNet-50 bottleneck topology
    and ResNeXt-style grouped widening: ``cardinality = 32``,
    ``base_width = 4``, ``split_input = False``.  The bottleneck
    width per stage follows the ResNeXt formula

    .. math::

        \text{width} = \lfloor \mathrm{planes} \cdot
            \tfrac{\text{base\_width}}{64} \rfloor \cdot
            \text{cardinality},

    matching the ``SKNet-50`` entry in Li et al., 2019.
    Approximately 27.5M parameters and 77.5% ImageNet-1k top-1
    accuracy in the paper.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNet
        Backbone with the SK-ResNeXt-50-32×4d configuration
        applied (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1 (SKNet-50 row).  Combines the
    cardinality of ResNeXt with the data-dependent receptive-field
    selection of SK.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnext_50_32x4d
    >>> model = sk_resnext_50_32x4d()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 7, 7)
    """
    cfg = (
        replace(_CFG_SK_RX50, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK_RX50
    )
    return SKNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


# reason: sk_resnet_18_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory
# protocol predates the v3.1 weights system and still names only pretrained + **overrides.
@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK18,
)
def sk_resnet_18_cls(
    pretrained: bool | str = False,
    *,
    weights: SKResNet18Weights | None = None,
    **overrides: object,
) -> SKNetForImageClassification:
    r"""SK-ResNet-18 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-18 backbone (basic blocks stacked ``[2, 2, 2, 2]``;
    the first :math:`3 \times 3` of each block is a Selective Kernel
    unit, the second a plain conv) followed by global average pooling
    and a linear projection to ``config.num_classes``.  Approximately
    11.96M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SKResNet18Weights.RA_IN1K`); a
        tag string (e.g. ``"RA_IN1K"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both are
        given).
    weights : SKResNet18Weights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``SKResNet18Weights.RA_IN1K``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-18 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).  Pretrained weights are converted from
    ``timm``'s ``skresnet18.ra_in1k`` and hosted on the Hugging Face
    Hub under ``lucid-dl/sk-resnet-18``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_18_cls
    >>> model = sk_resnet_18_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = sk_resnet_18_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.vision.sknet import SKResNet18Weights
    >>> model = sk_resnet_18_cls(weights=SKResNet18Weights.RA_IN1K)
    """
    entry = weights_mod.resolve_weights(SKResNet18Weights, pretrained, weights)
    cfg = (
        replace(_CFG_SK18, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK18
    )
    model = SKNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="sk_resnet_18_cls")
    return model


# reason: sk_resnet_34_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory
# protocol predates the v3.1 weights system and still names only pretrained + **overrides.
@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK34,
)
def sk_resnet_34_cls(
    pretrained: bool | str = False,
    *,
    weights: SKResNet34Weights | None = None,
    **overrides: object,
) -> SKNetForImageClassification:
    r"""SK-ResNet-34 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-34 backbone (basic blocks stacked ``[3, 4, 6, 3]``;
    the first :math:`3 \times 3` of each block is a Selective Kernel
    unit, the second a plain conv) followed by global average pooling
    and a linear projection.  Approximately 22.28M parameters.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`SKResNet34Weights.RA_IN1K`); a
        tag string → that specific checkpoint.  Mutually exclusive with
        ``weights`` (which wins if both are given).
    weights : SKResNet34Weights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-34 configuration applied
        (or with ``overrides`` merged on top of it), optionally
        initialised from pretrained weights.

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).  Pretrained weights are converted from
    ``timm``'s ``skresnet34.ra_in1k`` and hosted on the Hugging Face
    Hub under ``lucid-dl/sk-resnet-34``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_34_cls
    >>> model = sk_resnet_34_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    entry = weights_mod.resolve_weights(SKResNet34Weights, pretrained, weights)
    cfg = (
        replace(_CFG_SK34, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK34
    )
    model = SKNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="sk_resnet_34_cls")
    return model


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK50,
)
def sk_resnet_50_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    r"""SK-ResNet-50 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-50 backbone (bottleneck blocks stacked
    ``[3, 4, 6, 3]``, one SK unit per bottleneck) followed by
    global average pooling and a linear projection.  Approximately
    25.8M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-50 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_50_cls
    >>> model = sk_resnet_50_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    cfg = (
        replace(_CFG_SK50, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK50
    )
    return SKNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK101,
)
def sk_resnet_101_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    r"""SK-ResNet-101 image classifier (backbone + GAP + linear head).

    Builds an :class:`SKNetForImageClassification` with the
    SK-ResNet-101 backbone (bottleneck blocks stacked
    ``[3, 4, 23, 3]``) followed by global average pooling and a
    linear classifier.  Approximately 45M parameters.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNet-101 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnet_101_cls
    >>> model = sk_resnet_101_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = (
        replace(_CFG_SK101, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK101
    )
    return SKNetForImageClassification(cfg)


@register_model(
    task="image-classification",
    family="sknet",
    model_type="sknet",
    model_class=SKNetForImageClassification,
    default_config=_CFG_SK_RX50,
)
def sk_resnext_50_32x4d_cls(
    pretrained: bool = False, **overrides: object
) -> SKNetForImageClassification:
    r"""SK-ResNeXt-50 32×4d image classifier — the paper's SKNet-50.

    Builds an :class:`SKNetForImageClassification` with the
    ResNeXt-style SK backbone (``cardinality = 32``,
    ``base_width = 4``) followed by global average pooling and a
    linear projection to ``config.num_classes``.  Approximately
    27.5M parameters and 77.5% ImageNet-1k top-1 accuracy in
    Li et al., 2019 (Table 1, SKNet-50 row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`SKNetConfig`.

    Returns
    -------
    SKNetForImageClassification
        Classifier with the SK-ResNeXt-50-32×4d configuration
        applied (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Li et al., "Selective Kernel Networks", CVPR 2019
    (arXiv:1903.06586), Table 1 (SKNet-50 row).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.sknet import sk_resnext_50_32x4d_cls
    >>> model = sk_resnext_50_32x4d_cls()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = (
        replace(_CFG_SK_RX50, **cast(dict[str, Any], overrides))
        if overrides
        else _CFG_SK_RX50
    )
    return SKNetForImageClassification(cfg)
