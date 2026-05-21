"""Registry factories for CSPResNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.cspnet._config import CSPNetConfig
from lucid.models.vision.cspnet._model import CSPNet, CSPNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_50 = CSPNetConfig(layers=(3, 3, 5, 2), channels=(64, 128, 256, 512))

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNet,
    default_config=_CFG_50,
)
def cspresnet_50(pretrained: bool = False, **overrides: object) -> CSPNet:
    r"""CSPResNet-50 feature-extracting backbone (no classification head).

    Builds a :class:`CSPNet` with ResNet-50-style bottleneck
    topology wrapped in the Cross-Stage-Partial transformation:
    block repeats ``(3, 3, 5, 2)`` over four stages at channel
    widths ``(64, 128, 256, 512)``.  Approximately 21.6M
    parameters.  Wang et al., 2020 report 76.6% ImageNet-1k
    top-1 accuracy — comparable to plain ResNet-50 (76.1%) but at
    substantially reduced FLOPs and memory.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`CSPNetConfig`
        (e.g. ``in_channels=1`` for grayscale input).

    Returns
    -------
    CSPNet
        Backbone with the CSPResNet-50 configuration applied (or
        with ``overrides`` merged on top of it).

    Notes
    -----
    See Wang et al., "CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN", CVPR Workshops 2020
    (arXiv:1911.11929).  The key idea is the channel-wise split
    that halves the FLOPs of each residual stage while
    diversifying the gradient paths through truncated dense
    connections.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspresnet_50
    >>> model = cspresnet_50()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape
    (1, 512, 7, 7)
    """
    cfg = CSPNetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return CSPNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="cspnet",
    model_type="cspnet",
    model_class=CSPNetForImageClassification,
    default_config=_CFG_50,
)
def cspresnet_50_cls(
    pretrained: bool = False, **overrides: object
) -> CSPNetForImageClassification:
    r"""CSPResNet-50 image classifier (backbone + GAP + linear head).

    Builds a :class:`CSPNetForImageClassification` with the
    canonical CSPResNet-50 backbone followed by global average
    pooling and a linear projection to ``config.num_classes``
    (default 1000 for ImageNet-1k).  Approximately 21.6M
    parameters and 76.6% ImageNet-1k top-1 accuracy (Wang et al.,
    2020).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`CSPNetConfig`
        (typically ``num_classes`` to retarget the classifier).

    Returns
    -------
    CSPNetForImageClassification
        Classifier with the CSPResNet-50 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Wang et al., "CSPNet: A New Backbone that can Enhance
    Learning Capability of CNN", CVPR Workshops 2020
    (arXiv:1911.11929).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.cspnet import cspresnet_50_cls
    >>> model = cspresnet_50_cls(num_classes=10)
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 10)
    """
    cfg = CSPNetConfig(**{**_CFG_50.__dict__, **overrides}) if overrides else _CFG_50
    return CSPNetForImageClassification(cfg)
