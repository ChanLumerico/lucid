"""Registry factories for MaskFormer variants."""

from lucid.models._registry import register_model
from lucid.models.vision.maskformer._config import MaskFormerConfig
from lucid.models.vision.maskformer._model import MaskFormerForSemanticSegmentation

_CFG_R50 = MaskFormerConfig(
    num_classes=150,
    in_channels=3,
    backbone_layers=(3, 4, 6, 3),
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    fpn_out_channels=256,
)

_CFG_R101 = MaskFormerConfig(
    num_classes=150,
    in_channels=3,
    backbone_layers=(3, 4, 23, 3),  # ResNet-101
    d_model=256,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.1,
    num_queries=100,
    fpn_out_channels=256,
)


def _build(
    cfg: MaskFormerConfig, kw: dict[str, object]
) -> MaskFormerForSemanticSegmentation:
    return MaskFormerForSemanticSegmentation(
        MaskFormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="semantic-segmentation",
    family="maskformer",
    model_type="maskformer",
    model_class=MaskFormerForSemanticSegmentation,
    default_config=_CFG_R50,
)
def maskformer_resnet50(
    pretrained: bool = False,
    **overrides: object,
) -> MaskFormerForSemanticSegmentation:
    r"""MaskFormer with ResNet-50 backbone (Cheng et al., NeurIPS 2021).

    Builds a :class:`MaskFormerForSemanticSegmentation` with the paper-cited
    ResNet-50 configuration: 100 segmentation queries, a 6-layer transformer
    decoder with ``d_model = 256``, 8 attention heads, ``dim_feedforward =
    2048``, and a 256-channel FPN pixel decoder.  Default targets ADE20K
    (150 semantic classes); reaches ADE20K validation mIoU of 44.5%
    (paper Table 3, MaskFormer-R50 row) — competitive with much heavier
    per-pixel baselines.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`MaskFormerConfig`
        (``num_classes``, ``num_queries``, ``d_model``,
        ``num_decoder_layers``, ``dropout``, ``fpn_out_channels``).

    Returns
    -------
    MaskFormerForSemanticSegmentation
        Segmentation model with the MaskFormer-R50 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Cheng et al., "Per-Pixel Classification is Not All You Need for
    Semantic Segmentation", NeurIPS 2021 (arXiv:2107.06278).  The
    unification of semantic / instance / panoptic under mask
    classification is the conceptual key idea.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maskformer import maskformer_resnet50
    >>> model = maskformer_resnet50(num_classes=21)
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    22
    """
    return _build(_CFG_R50, overrides)


@register_model(
    task="semantic-segmentation",
    family="maskformer",
    model_type="maskformer",
    model_class=MaskFormerForSemanticSegmentation,
    default_config=_CFG_R101,
)
def maskformer_resnet101(
    pretrained: bool = False,
    **overrides: object,
) -> MaskFormerForSemanticSegmentation:
    r"""MaskFormer with ResNet-101 backbone (Cheng et al., NeurIPS 2021).

    Builds a :class:`MaskFormerForSemanticSegmentation` with the same
    transformer head as :func:`maskformer_resnet50` but a deeper ResNet-101
    backbone (``[3, 4, 23, 3]`` bottleneck blocks).  Reaches ADE20K
    validation mIoU of 45.5% (paper Table 3, MaskFormer-R101 row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`MaskFormerConfig`.

    Returns
    -------
    MaskFormerForSemanticSegmentation
        Segmentation model with the MaskFormer-R101 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Cheng et al., "Per-Pixel Classification is Not All You Need for
    Semantic Segmentation", NeurIPS 2021 (arXiv:2107.06278).  Switching
    backbones is the only change versus :func:`maskformer_resnet50`; all
    transformer-head hyperparameters are shared.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.maskformer import maskformer_resnet101
    >>> model = maskformer_resnet101()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    151
    """
    return _build(_CFG_R101, overrides)
