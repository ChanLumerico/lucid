"""Registry factories for GoogLeNet."""

from dataclasses import replace
from typing import Any, cast

import lucid.weights as weights_mod
from lucid.models._registry import register_model
from lucid.models.vision.googlenet._config import GoogLeNetConfig
from lucid.models.vision.googlenet._model import (
    GoogLeNet,
    GoogLeNetForImageClassification,
)
from lucid.models.vision.googlenet._weights import GoogLeNetWeights

_CFG = GoogLeNetConfig()
_CFG_NO_AUX = GoogLeNetConfig(aux_logits=False)


@register_model(
    task="base",
    family="googlenet",
    model_type="googlenet",
    model_class=GoogLeNet,
    default_config=_CFG,
)
def googlenet(pretrained: bool = False, **overrides: object) -> GoogLeNet:
    r"""GoogLeNet (Inception v1) feature-extracting backbone.

    Builds a :class:`GoogLeNet` with the paper-cited Szegedy 2015
    topology: a Conv-MaxPool stem followed by nine
    :class:`_InceptionModule` blocks at three resolutions
    (28×28 → 14×14 → 7×7) and a final
    :class:`~lucid.nn.AdaptiveAvgPool2d` to :math:`1\times1`.
    Approximately 6.8 M parameters in the backbone — roughly **12×
    fewer** than AlexNet despite being substantially deeper.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`GoogLeNetConfig`
        (``in_channels``, etc.).  Auxiliary-classifier fields are
        irrelevant for the backbone.

    Returns
    -------
    GoogLeNet
        Backbone with the GoogLeNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Going Deeper with Convolutions", CVPR 2015 —
    the ILSVRC-2014 classification winner with a top-5 ImageNet
    validation error of 6.67%.  Single architecture; no paper-cited
    "tiny / large" variants (H11).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.googlenet import googlenet
    >>> model = googlenet()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape   # (B, 1024, 1, 1)
    (1, 1024, 1, 1)
    """
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    return GoogLeNet(cfg)


@register_model(  # type: ignore[arg-type]  # reason: googlenet_cls adds a typed weights= kwarg (GoogLeNetWeights); the ModelFactory protocol predates the v3.1 weights system and still names only pretrained + **overrides.
    task="image-classification",
    family="googlenet",
    model_type="googlenet",
    model_class=GoogLeNetForImageClassification,
    default_config=_CFG,
)
def googlenet_cls(
    pretrained: bool | str = False,
    *,
    weights: GoogLeNetWeights | None = None,
    **overrides: object,
) -> GoogLeNetForImageClassification:
    r"""GoogLeNet (Inception v1) image classifier with auxiliary heads.

    Builds a :class:`GoogLeNetForImageClassification` with the
    paper-cited Szegedy 2015 configuration: 22-layer Inception backbone
    + global-average-pool + dropout (``p=0.4``) + linear projection to
    ``config.num_classes``, plus two auxiliary classifiers attached at
    Inception 4a and 4d (enabled by default via ``aux_logits=True``).
    Approximately 13.0 M parameters total when auxiliary heads are
    included.  Reaches **69.778% top-1 on ImageNet-1k**.

    Parameters
    ----------
    pretrained : bool or str, optional, default=False
        Pretrained-weight selector.  ``False`` → random init; ``True``
        → the ``DEFAULT`` tag (:attr:`GoogLeNetWeights.IMAGENET1K_V1`);
        a tag string → that specific checkpoint.  Mutually exclusive
        with ``weights`` (which wins if both are given).
    weights : GoogLeNetWeights, optional, keyword-only
        Explicit weights enum member.  Takes precedence over
        ``pretrained``.
    **overrides
        Keyword overrides forwarded into :class:`GoogLeNetConfig`.  Use
        ``aux_logits=False`` to disable auxiliary classifiers for
        cheaper inference graphs, ``num_classes=N`` to retarget the
        head, or ``dropout=p`` / ``aux_dropout=p`` to adjust
        regularisation.

    Returns
    -------
    GoogLeNetForImageClassification
        Classifier with the GoogLeNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Going Deeper with Convolutions", CVPR 2015,
    §5.  Auxiliary classifiers were introduced specifically to combat
    vanishing gradients in this 22-layer network without the benefit of
    residual connections; they contribute to the loss with weight
    :math:`0.3` each during training and are discarded at inference.
    Pretrained weights are converted from torchvision's
    ``GoogLeNet_Weights.IMAGENET1K_V1`` and hosted under
    ``lucid-dl/googlenet``.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.googlenet import googlenet_cls
    >>> model = googlenet_cls().eval()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """
    entry = weights_mod.resolve_weights(GoogLeNetWeights, pretrained, weights)
    cfg = replace(_CFG, **cast(dict[str, Any], overrides)) if overrides else _CFG
    model = GoogLeNetForImageClassification(cfg)
    if entry is not None:
        weights_mod.load_weight_entry(model, entry, name="googlenet_cls")
    return model
