"""Registry factories for Attention U-Net variants."""

from lucid.models._registry import register_model
from lucid.models.vision.attention_unet._config import AttentionUNetConfig
from lucid.models.vision.attention_unet._model import (
    AttentionUNetForSemanticSegmentation,
)

_CFG_BASE = AttentionUNetConfig(
    num_classes=2,
    in_channels=1,
    base_channels=64,
    depth=4,
    bilinear=False,
)


def _build(
    cfg: AttentionUNetConfig, kw: dict[str, object]
) -> AttentionUNetForSemanticSegmentation:
    return AttentionUNetForSemanticSegmentation(
        AttentionUNetConfig(**{**cfg.__dict__, **kw}) if kw else cfg
    )


@register_model(
    task="semantic-segmentation",
    family="attention_unet",
    model_type="attention_unet",
    model_class=AttentionUNetForSemanticSegmentation,
    default_config=_CFG_BASE,
)
def attention_unet(
    pretrained: bool = False,
    **overrides: object,
) -> AttentionUNetForSemanticSegmentation:
    """Attention U-Net (Oktay et al., MIDL 2018).

    Standard configuration: 4-level encoder/decoder, base_channels=64,
    in_channels=1 (medical imaging default), 2 output classes.
    Soft attention gates suppress irrelevant skip-connection features.
    """
    return _build(_CFG_BASE, overrides)
