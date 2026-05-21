"""Registry factories for Inception-ResNet v2."""

from lucid.models._registry import register_model
from lucid.models.vision.inception_resnet._config import InceptionResNetConfig
from lucid.models.vision.inception_resnet._model import (
    InceptionResNetV2,
    InceptionResNetV2ForImageClassification,
)

_CFG = InceptionResNetConfig()


@register_model(
    task="base",
    family="inception_resnet",
    model_type="inception_resnet",
    model_class=InceptionResNetV2,
    default_config=_CFG,
)
def inception_resnet_v2(
    pretrained: bool = False, **overrides: object
) -> InceptionResNetV2:
    r"""Inception-ResNet v2 feature-extracting backbone.

    Builds an :class:`InceptionResNetV2` with the paper-cited Szegedy
    2017 topology: 5-conv stem → Mixed_5b → 10× Block35 → Reduction-A
    → 20× Block17 → Reduction-B → 9× Block8 (+ 1 no-ReLU block) →
    :math:`1\times1` projection to 1536 channels.  Designed for
    :math:`299\times299` RGB inputs.  Approximately 54.3 M parameters
    in the backbone alone.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored — the returned model is randomly initialised.
    **overrides
        Keyword overrides forwarded into :class:`InceptionResNetConfig`.
        Use ``scale_a`` / ``scale_b`` / ``scale_c`` to tune the
        residual-branch scaling factors (defaults 0.17 / 0.10 / 0.20
        from the paper).

    Returns
    -------
    InceptionResNetV2
        Backbone with the Inception-ResNet v2 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact
    of Residual Connections on Learning", AAAI 2017.  Top-5 ImageNet
    validation error of 3.08% (full classifier).  No paper-cited size
    variants (H11) — only v2 is implemented here; v1 differs primarily
    in widths.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_resnet import inception_resnet_v2
    >>> model = inception_resnet_v2()
    >>> x = lucid.randn(1, 3, 299, 299)
    >>> out = model(x)
    >>> out.last_hidden_state.shape   # (B, 1536, 1, 1)
    (1, 1536, 1, 1)
    """
    cfg = InceptionResNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionResNetV2(cfg)


@register_model(
    task="image-classification",
    family="inception_resnet",
    model_type="inception_resnet",
    model_class=InceptionResNetV2ForImageClassification,
    default_config=_CFG,
)
def inception_resnet_v2_cls(
    pretrained: bool = False, **overrides: object
) -> InceptionResNetV2ForImageClassification:
    r"""Inception-ResNet v2 image classifier (backbone + GAP + dropout + linear).

    Builds an :class:`InceptionResNetV2ForImageClassification` with the
    paper-cited Szegedy 2017 topology and a single
    :class:`~lucid.nn.Linear` classifier head producing
    ``config.num_classes`` logits.  Approximately 55.8 M parameters
    total; achieves a top-5 ImageNet validation error of 3.08%.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently
        ignored.
    **overrides
        Keyword overrides forwarded into :class:`InceptionResNetConfig`.
        Common picks: ``num_classes=N`` to retarget the head,
        ``dropout=p`` to adjust regularisation.

    Returns
    -------
    InceptionResNetV2ForImageClassification
        Classifier with the Inception-ResNet v2 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Szegedy et al., "Inception-v4, Inception-ResNet and the Impact
    of Residual Connections on Learning", AAAI 2017, §3.3.  The
    classifier attribute is named ``classif`` (not ``classifier``) for
    timm / TensorFlow-Slim state-dict compatibility.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.inception_resnet import inception_resnet_v2_cls
    >>> model = inception_resnet_v2_cls()
    >>> x = lucid.randn(2, 3, 299, 299)
    >>> out = model(x)
    >>> out.logits.shape
    (2, 1000)
    """
    cfg = InceptionResNetConfig(**{**_CFG.__dict__, **overrides}) if overrides else _CFG
    return InceptionResNetV2ForImageClassification(cfg)
