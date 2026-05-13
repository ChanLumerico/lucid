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

_CFG_SMALL = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=32,
    depth=3,
    bilinear=False,
    dropout=0.0,
)

_CFG_BILINEAR = UNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=64,
    depth=4,
    bilinear=True,
    dropout=0.0,
)


def _build(
    cfg: UNetConfig, kw: dict[str, object]
) -> UNetForSemanticSegmentation:
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
    default_config=_CFG_SMALL,
)
def unet_small(
    pretrained: bool = False,
    **overrides: object,
) -> UNetForSemanticSegmentation:
    """U-Net — small variant.

    Lightweight configuration: 3-level encoder/decoder, base_channels=32.
    Suitable for rapid experimentation or memory-constrained settings.
    """
    return _build(_CFG_SMALL, overrides)


@register_model(
    task="semantic-segmentation",
    family="unet",
    model_type="unet",
    model_class=UNetForSemanticSegmentation,
    default_config=_CFG_BILINEAR,
)
def unet_bilinear(
    pretrained: bool = False,
    **overrides: object,
) -> UNetForSemanticSegmentation:
    """U-Net — bilinear upsampling variant.

    Same as standard U-Net but uses bilinear interpolation + Conv2d for
    upsampling in the decoder instead of ConvTranspose2d.  Reduces the
    total parameter count slightly.
    """
    return _build(_CFG_BILINEAR, overrides)
