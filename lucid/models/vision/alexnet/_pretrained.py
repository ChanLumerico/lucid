"""Registry factories for AlexNet."""

from lucid.models._registry import register_model
from lucid.models.vision.alexnet._config import AlexNetConfig
from lucid.models.vision.alexnet._model import AlexNet, AlexNetForImageClassification

_CFG = AlexNetConfig()


@register_model(
    task="base",
    family="alexnet",
    model_type="alexnet",
    model_class=AlexNet,
    default_config=_CFG,
    params=3_747_200,  # conv trunk only (Krizhevsky et al. 2012)
)
def alexnet(pretrained: bool = False, **overrides: object) -> AlexNet:
    r"""AlexNet feature-extracting backbone (no fully-connected head).

    Builds an :class:`AlexNet` with the paper-cited Krizhevsky 2012
    topology: five convolutional blocks (Conv1: 3→96 with
    :math:`11\times11` stride-4; Conv2: 96→256 with :math:`5\times5`;
    three :math:`3\times3` convolutions Conv3-5 with widths 384/384/256),
    :class:`~lucid.nn.LocalResponseNorm` after blocks 1 and 2, and
    overlapping :math:`3\times3` max-pools after blocks 1, 2, 5.
    Approximately 3.7 M parameters in the convolutional trunk alone
    (≈60 M for the full classifier variant).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`AlexNetConfig` to
        customise individual fields (``in_channels=1`` for grayscale,
        ``dropout=0.0`` to disable the dropout layers of the
        classifier head).

    Returns
    -------
    AlexNet
        Backbone with the AlexNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Krizhevsky et al., "ImageNet Classification with Deep
    Convolutional Neural Networks", NIPS 2012, §3.  The original paper
    reports a 15.3% ImageNet-1k top-5 validation error — more than ten
    percentage points below the runner-up — and is widely credited with
    re-igniting deep-learning research after the 2006-2011 lull.  No
    paper-cited "small / large" variants exist (H11) — AlexNet is a
    single architecture.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet
    >>> model = alexnet()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 256, 6, 6)
    (1, 256, 6, 6)
    """
    cfg = AlexNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return AlexNet(cfg)


@register_model(
    task="image-classification",
    family="alexnet",
    model_type="alexnet",
    model_class=AlexNetForImageClassification,
    default_config=_CFG,
    params=61_100_840,  # full classifier (Krizhevsky et al. 2012)
)
def alexnet_cls(
    pretrained: bool = False, **overrides: object
) -> AlexNetForImageClassification:
    r"""AlexNet image classifier (backbone + FC6 + FC7 + linear head).

    Builds an :class:`AlexNetForImageClassification` with the
    paper-cited Krizhevsky 2012 topology: the five-block conv trunk
    followed by FC6 (9216 → 4096), FC7 (4096 → 4096), and a final
    linear projection to ``config.num_classes`` (default 1000 for
    ImageNet-1k).  Approximately 60 M parameters total, dominated by
    the two 4096-dim FC layers (≈54 M).  :class:`~lucid.nn.Dropout`
    with ``p=config.dropout`` (default 0.5) is applied after both
    hidden activations.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`AlexNetConfig`.  Use
        ``num_classes=N`` to retarget the classifier (e.g.
        ``num_classes=10`` for CIFAR-10 fine-tuning) and ``dropout=p``
        to adjust the regularisation strength.

    Returns
    -------
    AlexNetForImageClassification
        Classifier with the AlexNet configuration applied (or with
        ``overrides`` merged on top of it).

    Notes
    -----
    See Krizhevsky et al., "ImageNet Classification with Deep
    Convolutional Neural Networks", NIPS 2012, §3.  The original paper
    reports 15.3% ImageNet-1k top-5 validation error.  The two
    :class:`~lucid.nn.Dropout` layers were a key empirical contribution
    of the paper: setting half of each 4096-dim activation to zero
    during training was the principal technique that prevented
    overfitting of a 60 M-parameter network on 1.2 M images.

    Examples
    --------
    Run a forward pass without labels:

    >>> import lucid
    >>> from lucid.models.vision.alexnet import alexnet_cls
    >>> model = alexnet_cls()
    >>> x = lucid.randn(2, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)

    Retarget to CIFAR-10:

    >>> model = alexnet_cls(num_classes=10)
    >>> model.config.num_classes
    10
    """
    cfg = AlexNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return AlexNetForImageClassification(cfg)
