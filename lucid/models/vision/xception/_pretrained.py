"""Registry factories for Xception."""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.xception._config import XceptionConfig
from lucid.models.vision.xception._model import (
    Xception,
    XceptionForImageClassification,
)
from lucid.models.vision.xception._weights import XceptionWeights

_CFG = XceptionConfig()


@register_model(
    task="base",
    family="xception",
    model_type="xception",
    model_class=Xception,
    default_config=_CFG,
)
def xception(pretrained: bool = False, **overrides: object) -> Xception:
    r"""Xception feature-extracting backbone (no classification head).

    Builds an :class:`Xception` with the paper-cited "Extreme
    Inception" topology from Chollet, 2017: a 2-conv stem, three
    entry-flow blocks (1×1 strided residuals with channel
    progression 64 → 128 → 256 → 728), eight middle-flow blocks at
    728 channels, one exit-flow transition block (728 → 1024),
    and two final separable convolutions expanding to 1536 → 2048
    channels.  Approximately 22.9M parameters — the same budget as
    Inception-v3 but with depthwise separable convolutions
    throughout.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`XceptionConfig`
        (e.g. ``in_channels=1`` for grayscale input).

    Returns
    -------
    Xception
        Backbone with the canonical Xception configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Chollet, "Xception: Deep Learning with Depthwise Separable
    Convolutions", CVPR 2017 (arXiv:1610.02357).  Designed for
    299×299 input — the same crop as Inception-v3 — so the
    spatial-reduction schedule isolates the depthwise-separable
    factorisation as the sole source of accuracy improvement.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.xception import xception
    >>> model = xception()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 2048, 1, 1)
    """
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    return Xception(cfg)


@register_model(  # type: ignore[arg-type]  # reason: xception_cls adds typed weights= kwarg (per-model WeightsEnum); ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="xception",
    model_type="xception",
    model_class=XceptionForImageClassification,
    default_config=_CFG,
)
def xception_cls(
    pretrained: bool | str = False,
    *,
    weights: XceptionWeights | None = None,
    **overrides: object,
) -> XceptionForImageClassification:
    r"""Xception image classifier (backbone + dropout + linear head).

    Builds an :class:`XceptionForImageClassification` with the
    canonical Xception backbone (Chollet, 2017) followed by
    dropout (:math:`p = 0.5`) and a linear projection to
    ``config.num_classes`` (default 1000 for ImageNet-1k).
    Approximately 22.9M parameters and 79.0% ImageNet-1k top-1
    accuracy at 299×299 (paper, Table 5) — outperforming
    Inception-v3 at the same parameter budget.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`XceptionWeights.TF_IN1K`); a tag
        string (e.g. ``"TF_IN1K"``) → that specific checkpoint.
        Mutually exclusive with ``weights`` (which wins if both are
        given).
    weights : XceptionWeights, optional, keyword-only
        Explicit weights enum member, e.g.
        ``XceptionWeights.TF_IN1K``.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`XceptionConfig`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    XceptionForImageClassification
        Classifier with the canonical Xception configuration
        applied (or with ``overrides`` merged on top of it),
        optionally initialised from pretrained weights.

    Notes
    -----
    See Chollet, "Xception: Deep Learning with Depthwise Separable
    Convolutions", CVPR 2017 (arXiv:1610.02357), Table 5.  Pretrained
    weights are converted from timm's ``legacy_xception.tf_in1k`` and
    hosted on the Hugging Face Hub under ``lucid-dl/xception``.  The
    eval preset is 299 crop / 333 resize / bicubic / mean=std=0.5 (not
    ImageNet stats).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.xception import xception_cls
    >>> model = xception_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)

    Load ImageNet-pretrained weights:

    >>> model = xception_cls(pretrained=True)            # DEFAULT tag
    >>> from lucid.models.vision.xception import XceptionWeights
    >>> model = xception_cls(weights=XceptionWeights.TF_IN1K)
    """
    entry = weights_mod.resolve_weights(XceptionWeights, pretrained, weights)
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    model = XceptionForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="xception_cls")
    return model
