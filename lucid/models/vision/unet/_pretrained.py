"""Registry factories for U-Net variants."""

from lucid.models._registry import register_model
from lucid.models.vision.unet._config import UNetConfig
from lucid.models.vision.unet._model import UNetForSemanticSegmentation

_CFG_BASE = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=64,
    depth=4,
    bilinear=False,
    dropout=0.0,
)

# Residual variants (ResUNet)
_CFG_RES_2D = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=64,
    depth=4,
    bilinear=False,
    dropout=0.0,
    block="res",
)

# 3-D variants (volumetric segmentation)
_CFG_3D = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=32,
    depth=3,
    bilinear=False,
    dropout=0.0,
    dim=3,
)

_CFG_RES_3D = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=32,
    depth=3,
    bilinear=False,
    dropout=0.0,
    dim=3,
    block="res",
)


def _build(cfg: UNetConfig, kw: dict[str, object]) -> UNetForSemanticSegmentation:
    return UNetForSemanticSegmentation(
        UNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_BASE,
)
def unet(
    pretrained: bool = False,
    **overrides: object,
) -> UNetForSemanticSegmentation:
    """U-Net (Ronneberger et al., MICCAI 2015).

    Standard configuration: 4-level encoder/decoder, base_channels=64,
    in_channels=1 (biomedical imaging default), 2 output classes.
    ConvTranspose2d upsampling in the decoder.
    """
    return _build(_CFG_BASE, overrides)


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_RES_2D,
)
def res_unet_2d(
    pretrained: bool = False,
    **overrides: object,
) -> UNetForSemanticSegmentation:
    """ResUNet — U-Net with residual DoubleConv blocks (2-D).

    Adds an identity shortcut inside each DoubleConv, easing gradient flow
    in deeper variants.  Same depth and channel schedule as the standard
    U-Net, but every DoubleConv is residual.
    """
    return _build(_CFG_RES_2D, overrides)


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_3D,
)
def unet_3d(
    pretrained: bool = False,
    **overrides: object,
) -> UNetForSemanticSegmentation:
    """3-D U-Net for volumetric segmentation.

    All Conv2d / BatchNorm2d / MaxPool2d ops are replaced with their 3-D
    counterparts; "bilinear" upsampling becomes "trilinear".  Default
    config uses base_channels=32 and depth=3 to keep memory tractable.
    """
    return _build(_CFG_3D, overrides)


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_RES_3D,
)
def res_unet_3d(
    pretrained: bool = False,
    **overrides: object,
) -> UNetForSemanticSegmentation:
    """3-D ResUNet — volumetric residual U-Net variant."""
    return _build(_CFG_RES_3D, overrides)
