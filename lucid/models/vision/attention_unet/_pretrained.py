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
    r"""Attention U-Net (Oktay et al., MIDL 2018).

    Builds an :class:`AttentionUNetForSemanticSegmentation` with the
    standard configuration: 4-level encoder / decoder, ``base_channels =
    64`` (channel schedule 64 -> 128 -> 256 -> 512 -> 1024),
    ``in_channels = 1`` (medical imaging default), and ``num_classes = 2``.
    Soft attention gates on every skip connection suppress irrelevant
    encoder activations.

    Parameters
    ----------
    pretrained : bool, optional, default=False
        Reserved for future pretrained-weight loading.  Currently ignored.
    **overrides
        Keyword overrides forwarded into :class:`AttentionUNetConfig`
        (``num_classes``, ``in_channels`` for RGB inputs,
        ``base_channels``, ``depth``, ``bilinear``).

    Returns
    -------
    AttentionUNetForSemanticSegmentation
        Segmentation model with the standard Attention U-Net
        configuration applied (or with ``overrides`` merged on top of it).

    Notes
    -----
    See Oktay et al., "Attention U-Net: Learning Where to Look for the
    Pancreas", MIDL 2018 (arXiv:1804.03999).  The defining attention-gate
    update is

    .. math::

        \hat{x}^\ell = \sigma\!\bigl(\psi^\top
            \tanh(W_x x^\ell + W_g g^\ell)\bigr) \odot x^\ell,

    where :math:`x^\ell` is the encoder feature at level :math:`\ell`
    and :math:`g^\ell` is the up-sampled decoder feature serving as the
    gating signal.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.attention_unet import attention_unet
    >>> model = attention_unet(num_classes=4, in_channels=3)
    >>> x = lucid.randn(1, 3, 256, 256)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 4, 256, 256)
    """
    return _build(_CFG_BASE, overrides)
