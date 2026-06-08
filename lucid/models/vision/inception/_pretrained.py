"""Registry factories for Inception v3."""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.inception._config import InceptionConfig
from lucid.models.vision.inception._model import (
    InceptionV3,
    InceptionV3ForImageClassification,
)
from lucid.models.vision.inception._weights import InceptionV3Weights

_CFG = InceptionConfig(aux_logits=False)
_CFG_AUX = InceptionConfig(aux_logits=True)


@register_model(
    task="base",
    family="inception",
    model_type="inception_v3",
    model_class=InceptionV3,
    default_config=_CFG,
)
def inception_v3(pretrained: bool = False, **overrides: object) -> InceptionV3:
    r"""Inception v3 feature-extracting backbone.

    Builds an :class:`InceptionV3` with the paper-cited Szegedy 2016
    topology: deep stem → 3× Inception-A → Reduction-A → 4× Inception-C
    → Reduction-B → 2× Inception-E → :math:`1\times1` global pool.
    Designed for :math:`299\times299` RGB inputs.  Approximately
    21.8 M parameters in the backbone alone.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`InceptionConfig`.
        Note that ``aux_logits`` only affects the classifier variant.

    Returns
    -------
    InceptionV3
        Backbone with the Inception v3 configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Rethinking the Inception Architecture for
    Computer Vision", CVPR 2016.  Top-5 ImageNet validation error in
    the paper is 3.5% (full classifier).  Single architecture; no
    paper-cited size variants (H11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception import inception_v3
    >>> model = inception_v3()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 2048, 1, 1)
    (1, 2048, 1, 1)
    """
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    return InceptionV3(cfg)


# reason: inception_v3_cls adds typed weights= kwarg (per-model WeightsEnum);
# ModelFactory protocol predates the v3.1 weights system and still names only
# pretrained + **overrides.
@register_model(  # type: ignore[arg-type]
    task="image-classification",
    family="inception",
    model_type="inception_v3",
    model_class=InceptionV3ForImageClassification,
    default_config=_CFG,
)
def inception_v3_cls(
    pretrained: bool | str = False,
    *,
    weights: InceptionV3Weights | None = None,
    **overrides: object,
) -> InceptionV3ForImageClassification:
    r"""Inception v3 image classifier (no auxiliary head by default).

    Builds an :class:`InceptionV3ForImageClassification` with the
    paper-cited Szegedy 2016 topology.  Default configuration matches
    the standard reference-framework setting of ``aux_logits=False`` —
    approximately 23.8 M parameters; enable the auxiliary classifier
    by passing ``aux_logits=True`` (adds ≈4 M parameters used only
    during training).  Designed for :math:`299\times299` RGB inputs;
    achieves a top-5 ImageNet validation error of 3.5%.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`InceptionV3Weights.IMAGENET1K_V1`);
        a tag string (e.g. ``"IMAGENET1K_V1"``) → that specific
        checkpoint.  Mutually exclusive with ``weights`` (which wins if
        both are given).
    weights : InceptionV3Weights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``InceptionV3Weights.IMAGENET1K_V1``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`InceptionConfig`.
        Common picks: ``num_classes=N`` to retarget the head,
        ``aux_logits=True`` to enable the auxiliary classifier,
        ``dropout=p`` to adjust regularisation strength.  Note:
        overriding ``num_classes`` (or ``aux_logits``) away from the
        checkpoint topology makes pretrained loading fail the strict
        key/shape check.

    Returns
    -------
    InceptionV3ForImageClassification
        Classifier with the Inception v3 configuration applied (or with
        ``overrides`` merged on top of it), optionally initialised from
        pretrained weights.

    Notes
    -----
    See Szegedy et al., "Rethinking the Inception Architecture for
    Computer Vision", CVPR 2016, §6.  Training with the auxiliary head
    adds a 0.4-weighted cross-entropy term as

    .. math::

        \mathcal{L} = \mathcal{L}_{\text{main}}
            + 0.4 \cdot \mathcal{L}_{\text{aux}}.

    Pretrained weights are converted from torchvision's
    ``Inception_V3_Weights`` (auxiliary head dropped) and hosted on the
    Hugging Face Hub under ``lucid-dl/inception-v3``.  They evaluate at a
    299-pixel centre crop resized from 342.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception import inception_v3_cls
    >>> model = inception_v3_cls().eval()
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)

    Load ImageNet-pretrained weights:

    >>> model = inception_v3_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.vision.inception import InceptionV3Weights
    >>> model = inception_v3_cls(weights=InceptionV3Weights.IMAGENET1K_V1)
    """
    entry = weights_mod.resolve_weights(InceptionV3Weights, pretrained, weights)
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    model = InceptionV3ForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="inception_v3_cls")
    return model
