"""Registry factories for Inception-v4."""

from dataclasses import replace
from typing import Any, cast

from lucid.models._registry import register_model
from lucid.models._utils._common import reject_unavailable_pretrained
from lucid.models.vision.inception_v4._config import InceptionV4Config
from lucid.models.vision.inception_v4._model import (
    InceptionV4,
    InceptionV4ForImageClassification,
)

_CFG = InceptionV4Config()


@register_model(
    task="base",
    family="inception_v4",
    model_type="inception_v4",
    model_class=InceptionV4,
    default_config=_CFG,
    params=41_142_816,
    summary="auto",
)
def inception_v4(pretrained: bool = False, **overrides: object) -> InceptionV4:
    r"""Inception-v4 feature-extracting backbone.

    Builds an :class:`InceptionV4` with the topology of Szegedy et al.
    2017, Figure 9: multi-branch stem → 4× Inception-A → Reduction-A →
    7× Inception-B → Reduction-B → 3× Inception-C, ending in an
    :math:`8\times8\times1536` feature map for a :math:`299\times299`
    input.  41.1 M parameters (the published classifier's 42.7 M less
    its 1536 → 1000 head).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No pretrained weights are published for this factory; ``True``
        raises :class:`NotImplementedError` rather than returning a
        randomly initialised model.
    **overrides
        Keyword overrides forwarded into :class:`InceptionV4Config`
        (e.g. ``in_channels``).

    Returns
    -------
    InceptionV4
        Backbone with the Inception-v4 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact of
    Residual Connections on Learning", AAAI 2017.  The paper defines a
    single network, so there are no size variants (H11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_v4 import inception_v4
    >>> model = inception_v4().eval()
    >>> out = model(lucid.randn(1, 3, 299, 299))
    >>> out.last_hidden_state.shape
    (1, 1536, 8, 8)
    """
    if pretrained:
        reject_unavailable_pretrained("inception_v4")
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    return InceptionV4(cfg)


@register_model(
    task="image-classification",
    family="inception_v4",
    model_type="inception_v4",
    model_class=InceptionV4ForImageClassification,
    default_config=_CFG,
    params=42_679_816,
    summary="auto",
)
def inception_v4_cls(
    pretrained: bool = False, **overrides: object
) -> InceptionV4ForImageClassification:
    r"""Inception-v4 image classifier (trunk + GAP + dropout + linear).

    Builds an :class:`InceptionV4ForImageClassification` with the paper's
    topology and head — global average pool, dropout with keep
    probability 0.8, and a :class:`~lucid.nn.Linear` layer producing
    ``config.num_classes`` logits.  42.7 M parameters, matching the
    released TF-Slim ImageNet checkpoint.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        No pretrained weights are published for this factory yet;
        ``True`` raises :class:`NotImplementedError` rather than
        returning a randomly initialised model.
    **overrides
        Keyword overrides forwarded into :class:`InceptionV4Config`.
        Common picks: ``num_classes=N`` to retarget the head,
        ``dropout=p`` to change the head regularisation.

    Returns
    -------
    InceptionV4ForImageClassification
        Classifier with the Inception-v4 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact of
    Residual Connections on Learning", AAAI 2017.  Single-crop ImageNet
    validation error reported in Table 2: 20.0% top-1 / 5.0% top-5.  The
    head is named ``last_linear`` so the TF-Slim checkpoint's keys load
    unchanged.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_v4 import inception_v4_cls
    >>> model = inception_v4_cls(num_classes=10).eval()
    >>> out = model(lucid.randn(2, 3, 299, 299))
    >>> out.logits.shape
    (2, 10)
    """
    if pretrained:
        reject_unavailable_pretrained("inception_v4_cls")
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    return InceptionV4ForImageClassification(cfg)
