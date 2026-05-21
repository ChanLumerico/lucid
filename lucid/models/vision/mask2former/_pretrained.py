"""Registry factories for Mask2Former variants."""

from lucid.models._registry import register_model
from lucid.models.vision.mask2former._config import Mask2FormerConfig
from lucid.models.vision.mask2former._model import Mask2FormerForSemanticSegmentation

# ---------------------------------------------------------------------------
# ResNet-backbone configs
# ---------------------------------------------------------------------------


def _resnet_cfg(layers: tuple[int, int, int, int], block: str) -> Mask2FormerConfig:
    return Mask2FormerConfig(
        num_classes=150,
        in_channels=3,
        backbone_layers=layers,
        backbone_block=block,  # type: ignore[arg-type]
        backbone_type="resnet",
        d_model=256,
        n_head=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_queries=100,
        fpn_out_channels=256,
        num_feature_levels=3,
    )


_CFG_R50 = _resnet_cfg((3, 4, 6, 3), "bottleneck")
_CFG_R101 = _resnet_cfg((3, 4, 23, 3), "bottleneck")


# ---------------------------------------------------------------------------
# Swin-backbone configs (Liu et al., 2021)
# ---------------------------------------------------------------------------


def _swin_cfg(
    embed_dim: int,
    depths: tuple[int, int, int, int],
    num_heads: tuple[int, int, int, int],
) -> Mask2FormerConfig:
    return Mask2FormerConfig(
        num_classes=150,
        in_channels=3,
        backbone_type="swin",
        swin_embed_dim=embed_dim,
        swin_depths=depths,
        swin_num_heads=num_heads,
        swin_window_size=7,
        d_model=256,
        n_head=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        num_queries=100,
        fpn_out_channels=256,
        num_feature_levels=3,
    )


_CFG_SWIN_TINY = _swin_cfg(96, (2, 2, 6, 2), (3, 6, 12, 24))
_CFG_SWIN_SMALL = _swin_cfg(96, (2, 2, 18, 2), (3, 6, 12, 24))
_CFG_SWIN_BASE = _swin_cfg(128, (2, 2, 18, 2), (4, 8, 16, 32))
_CFG_SWIN_LARGE = _swin_cfg(192, (2, 2, 18, 2), (6, 12, 24, 48))


# ---------------------------------------------------------------------------
# Build helper
# ---------------------------------------------------------------------------


def _build(
    cfg: Mask2FormerConfig, kw: dict[str, object]
) -> Mask2FormerForSemanticSegmentation:
    return Mask2FormerForSemanticSegmentation(
        Mask2FormerConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


# ---------------------------------------------------------------------------
# ResNet factories
# ---------------------------------------------------------------------------


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_R50,
)
def mask2former_resnet50(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with ResNet-50 backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the
    paper-cited ResNet-50 configuration: masked-attention decoder
    cycling through 3 FPN levels (P3 / P4 / P5), 100 queries,
    ``d_model = 256``, 6 decoder layers, 8 attention heads, and a
    256-channel pixel decoder.  Default targets ADE20K (150 classes);
    reaches ADE20K validation mIoU of 47.2% (paper Table 2,
    Mask2Former-R50 row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-R50 configuration applied
        (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Cheng et al., "Masked-attention Mask Transformer for Universal
    Image Segmentation", CVPR 2022 (arXiv:2112.01527).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_resnet50
    >>> model = mask2former_resnet50()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    151
    """
    return _build(_CFG_R50, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_R101,
)
def mask2former_resnet101(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with ResNet-101 backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the same
    transformer head as :func:`mask2former_resnet50` but a deeper
    ResNet-101 backbone (``[3, 4, 23, 3]`` bottleneck blocks).  Reaches
    ADE20K validation mIoU of 47.8% (paper Table 2, Mask2Former-R101 row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-R101 configuration applied
        (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_resnet101
    >>> model = mask2former_resnet101()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    151
    """
    return _build(_CFG_R101, overrides)


# ---------------------------------------------------------------------------
# Swin factories
# ---------------------------------------------------------------------------


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_TINY,
)
def mask2former_swin_tiny(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Tiny backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the
    Swin-Tiny backbone (``embed_dim = 96``, ``depths = (2, 2, 6, 2)``,
    ``num_heads = (3, 6, 12, 24)``, ``window_size = 7``) and the standard
    masked-attention transformer head.  Defined by Liu et al., "Swin
    Transformer", ICCV 2021 (arXiv:2103.14030).  Reaches ADE20K
    validation mIoU of 49.6% (paper Table 2, Mask2Former-Swin-T row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-T configuration
        applied (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_tiny
    >>> model = mask2former_swin_tiny()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    151
    """
    return _build(_CFG_SWIN_TINY, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_SMALL,
)
def mask2former_swin_small(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Small backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the
    Swin-Small backbone (``embed_dim = 96``, ``depths = (2, 2, 18, 2)``,
    ``num_heads = (3, 6, 12, 24)``).  Same transformer head as
    :func:`mask2former_swin_tiny`; ADE20K validation mIoU of 51.0%
    (paper Table 2, Mask2Former-Swin-S row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-S configuration
        applied (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_small
    >>> model = mask2former_swin_small()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    151
    """
    return _build(_CFG_SWIN_SMALL, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_BASE,
)
def mask2former_swin_base(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Base backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the
    Swin-Base backbone (``embed_dim = 128``, ``depths = (2, 2, 18, 2)``,
    ``num_heads = (4, 8, 16, 32)``).  ADE20K validation mIoU of 52.4%
    (paper Table 2, Mask2Former-Swin-B row).

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-B configuration
        applied (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_base
    >>> model = mask2former_swin_base()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    151
    """
    return _build(_CFG_SWIN_BASE, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_LARGE,
)
def mask2former_swin_large(
    pretrained: bool = False,
    **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    r"""Mask2Former with Swin-Large backbone (Cheng et al., CVPR 2022).

    Builds a :class:`Mask2FormerForSemanticSegmentation` with the largest
    Swin backbone (``embed_dim = 192``, ``depths = (2, 2, 18, 2)``,
    ``num_heads = (6, 12, 24, 48)``).  Reaches ADE20K validation mIoU of
    56.1% (paper Table 2, Mask2Former-Swin-L row) — the headline result
    of the paper.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`Mask2FormerConfig`.

    Returns
    -------
    Mask2FormerForSemanticSegmentation
        Segmentation model with the Mask2Former-Swin-L configuration
        applied (or with ``overrides`` merged on top of it).

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.mask2former import mask2former_swin_large
    >>> model = mask2former_swin_large()
    >>> x = lucid.randn(1, 3, 512, 512)
    >>> out = model(x)
    >>> out.logits.shape[1]
    151
    """
    return _build(_CFG_SWIN_LARGE, overrides)
