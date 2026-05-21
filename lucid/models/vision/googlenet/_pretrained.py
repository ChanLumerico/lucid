"""Registry factories for GoogLeNet."""

from lucid.models._registry import register_model
from lucid.models.vision.googlenet._config import GoogLeNetConfig
from lucid.models.vision.googlenet._model import (
    GoogLeNet,
    GoogLeNetForImageClassification,
)

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
    cfg = GoogLeNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return GoogLeNet(cfg)


@register_model(
    task="image-classification",
    family="googlenet",
    model_type="googlenet",
    model_class=GoogLeNetForImageClassification,
    default_config=_CFG,
)
def googlenet_cls(
    pretrained: bool = False, **overrides: object
) -> GoogLeNetForImageClassification:
    r"""GoogLeNet (Inception v1) image classifier with auxiliary heads.

    Builds a :class:`GoogLeNetForImageClassification` with the
    paper-cited Szegedy 2015 configuration: 22-layer Inception backbone
    + global-average-pool + dropout (``p=0.4``) + linear projection to
    ``config.num_classes``, plus two auxiliary classifiers attached at
    Inception 4a and 4d (enabled by default via ``aux_logits=True``).
    Approximately 13 M parameters total when auxiliary heads are
    included, ≈6.8 M without.  Achieves a top-5 ImageNet-1k validation
    error of 6.67%.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
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
    cfg = GoogLeNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return GoogLeNetForImageClassification(cfg)
