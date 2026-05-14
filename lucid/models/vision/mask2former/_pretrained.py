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


_CFG_R18 = _resnet_cfg((2, 2, 2, 2), "basic")
_CFG_R34 = _resnet_cfg((3, 4, 6, 3), "basic")
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


_CFG_SWIN_TINY  = _swin_cfg( 96, (2, 2,  6, 2), (3,  6, 12, 24))
_CFG_SWIN_SMALL = _swin_cfg( 96, (2, 2, 18, 2), (3,  6, 12, 24))
_CFG_SWIN_BASE  = _swin_cfg(128, (2, 2, 18, 2), (4,  8, 16, 32))
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
    default_config=_CFG_R18,
)
def mask2former_resnet18(
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with ResNet-18 backbone (BasicBlock, expansion 1)."""
    return _build(_CFG_R18, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_R34,
)
def mask2former_resnet34(
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with ResNet-34 backbone (BasicBlock, expansion 1)."""
    return _build(_CFG_R34, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_R50,
)
def mask2former_resnet50(
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with ResNet-50 backbone (Cheng et al., CVPR 2022).

    Masked-attention transformer decoder with multi-scale FPN cross-attention
    (3 levels: P3/P4/P5).  100 queries, d_model=256, 6-layer decoder,
    8 attention heads.  ADE20K default (150 classes).
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
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with ResNet-101 backbone (deeper Bottleneck variant)."""
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
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with Swin-Tiny backbone (embed_dim=96, depths=2/2/6/2)."""
    return _build(_CFG_SWIN_TINY, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_SMALL,
)
def mask2former_swin_small(
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with Swin-Small backbone (embed_dim=96, depths=2/2/18/2)."""
    return _build(_CFG_SWIN_SMALL, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_BASE,
)
def mask2former_swin_base(
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with Swin-Base backbone (embed_dim=128, depths=2/2/18/2)."""
    return _build(_CFG_SWIN_BASE, overrides)


@register_model(
    task="semantic-segmentation",
    family="mask2former",
    model_type="mask2former",
    model_class=Mask2FormerForSemanticSegmentation,
    default_config=_CFG_SWIN_LARGE,
)
def mask2former_swin_large(
    pretrained: bool = False, **overrides: object,
) -> Mask2FormerForSemanticSegmentation:
    """Mask2Former with Swin-Large backbone (embed_dim=192, depths=2/2/18/2)."""
    return _build(_CFG_SWIN_LARGE, overrides)
