"""Registry factories for CoAtNet variants."""

from lucid.models._registry import register_model
from lucid.models.vision.coatnet._config import CoAtNetConfig
from lucid.models.vision.coatnet._model import CoAtNet, CoAtNetForImageClassification

# ---------------------------------------------------------------------------
# Canonical configs
# ---------------------------------------------------------------------------

_CFG_0 = CoAtNetConfig(
    variant="coatnet_0",
    blocks_per_stage=(2, 3, 5, 2),
    dims=(96, 192, 384, 768),
    stem_width=64,
    attn_heads=(12, 24),
    mbconv_expand=4,
    head_hidden_size=768,
)

# ---------------------------------------------------------------------------
# Backbone registrations (task="base")
# ---------------------------------------------------------------------------


@register_model(
    task="base",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNet,
    default_config=_CFG_0,
)
def coatnet_0(pretrained: bool = False, **overrides: object) -> CoAtNet:
    r"""CoAtNet-0 backbone (Dai et al., 2021).

    Builds the canonical *CoAtNet-0* configuration:
    ``blocks_per_stage=(2, 3, 5, 2)``, ``dims=(96, 192, 384, 768)``,
    ``stem_width=64``, ``attn_heads=(12, 24)``,
    ``head_hidden_size=768``.  The first two stages are MBConv with
    expansion ratio 4 and the last two stages are relative-attention
    transformers with head dim 32.  Approximately **25.6M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available in the model zoo.  Defaults to ``False``.
    **overrides : object
        Keyword overrides applied on top of the canonical CoAtNet-0
        config.  Common examples: ``image_size=384``, ``in_channels=1``,
        ``head_hidden_size=None``.  Each override must match a field of
        :class:`CoAtNetConfig`.

    Returns
    -------
    CoAtNet
        A :class:`CoAtNet` backbone returning a
        :math:`(B, 768, H/32, W/32)` spatial feature map.

    Notes
    -----
    CoAtNet-0 reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Dai
    et al., 2021, Table 5).  See
    `arXiv:2106.04803 <https://arxiv.org/abs/2106.04803>`_.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_0
    >>> model = coatnet_0()
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape
    (1, 768, 7, 7)
    """
    cfg = CoAtNetConfig(**{**_CFG_0.__dict__, **overrides}) if overrides else _CFG_0
    return CoAtNet(cfg)


# ---------------------------------------------------------------------------
# Classification head registrations (task="image-classification")
# ---------------------------------------------------------------------------


@register_model(
    task="image-classification",
    family="coatnet",
    model_type="coatnet",
    model_class=CoAtNetForImageClassification,
    default_config=_CFG_0,
)
def coatnet_0_cls(
    pretrained: bool = False, **overrides: object
) -> CoAtNetForImageClassification:
    r"""CoAtNet-0 image classifier (Dai et al., 2021).

    Combines the :func:`coatnet_0` backbone with the standard CoAtNet
    head: global average pool → LayerNorm → ``Linear(768) + Tanh`` →
    final linear classifier.  Default output is ``num_classes=1000``
    (ImageNet-1k).  Approximately **25.6M parameters**.

    Parameters
    ----------
    pretrained : bool, optional
        If ``True``, loads ImageNet-1k pretrained weights when
        available.  Defaults to ``False``.
    **overrides : object
        Keyword overrides on top of the canonical CoAtNet-0 config.
        Commonly used: ``num_classes`` to switch label space,
        ``image_size`` to fine-tune at a different resolution.

    Returns
    -------
    CoAtNetForImageClassification
        Classifier returning :class:`ImageClassificationOutput` whose
        ``logits`` has shape ``(B, num_classes)``.

    Notes
    -----
    CoAtNet-0 reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Dai
    et al., 2021, Table 5).  Larger variants (CoAtNet-3 / -4) reach
    **84.5% / 85.8%** at 384x384 after JFT-3B pretraining
    (Table 1, headline result).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.coatnet import coatnet_0_cls
    >>> model = coatnet_0_cls(num_classes=1000)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """
    cfg = CoAtNetConfig(**{**_CFG_0.__dict__, **overrides}) if overrides else _CFG_0
    return CoAtNetForImageClassification(cfg)
