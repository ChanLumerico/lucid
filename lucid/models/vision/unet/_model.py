"""U-Net (Ronneberger et al., MICCAI 2015).

Paper: "U-Net: Convolutional Networks for Biomedical Image Segmentation"

Key innovation
--------------
Symmetric encoder-decoder with skip connections.  Each decoder stage
concatenates the upsampled decoder feature with the spatially matched
encoder feature (skip connection), preserving fine-grained detail that
would otherwise be lost in the bottleneck.

Architecture
------------
  Encoder (left side):
    depth stages of DoubleConv followed by MaxPool(2).
    Channels: [base, base×2, base×4, …, base×2^(depth-1)]

  Bottleneck:
    DoubleConv with channels base×2^depth (no pooling).

  Decoder (right side):
    depth stages, each:
      1. Upsample ×2 (interp or ConvTranspose)
      2. Pad to match skip-connection spatial size if needed
      3. Concatenate with encoder skip feature
      4. DoubleConv → output channels halved

  Head:
    Conv(base_channels, num_classes, 1)

Output: logits (B, num_classes, *spatial) — same spatial size as input.

The same code path supports both 2-D (``dim=2``) and 3-D (``dim=3``)
inputs.  In 3-D mode all ops switch to their Conv3d / BatchNorm3d /
MaxPool3d / ConvTranspose3d counterparts; "bilinear" upsampling
becomes "trilinear".  A "res" block style adds an identity shortcut
inside DoubleConv to give the ResUNet variants.

Losses (training)
-----------------
  Requires integer ``targets`` of shape (B, *spatial).
  Uses cross-entropy loss: F.cross_entropy(logits, targets).
"""

from typing import ClassVar, cast, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models.vision.unet._config import UNetConfig

# ---------------------------------------------------------------------------
# Per-dim factories
# ---------------------------------------------------------------------------


def _conv(
    dim: int, in_ch: int, out_ch: int, k: int = 3, padding: int = 1, bias: bool = False
) -> nn.Module:
    if dim == 3:
        return nn.Conv3d(in_ch, out_ch, k, padding=padding, bias=bias)
    return nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=bias)


def _bn(dim: int, ch: int) -> nn.Module:
    return nn.BatchNorm3d(ch) if dim == 3 else nn.BatchNorm2d(ch)


def _pool(dim: int) -> nn.Module:
    return nn.MaxPool3d(2, stride=2) if dim == 3 else nn.MaxPool2d(2, stride=2)


def _deconv(dim: int, in_ch: int, out_ch: int) -> nn.Module:
    if dim == 3:
        return nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2)
    return nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)


def _interp_mode(dim: int) -> str:
    return "trilinear" if dim == 3 else "bilinear"


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Module):
    """Two sequential Conv-BN-ReLU blocks with optional dropout.

    When ``residual`` is True an identity shortcut (with optional 1×1
    channel projection) is added to the output, giving a ResUNet block.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dim: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self._dim = dim
        self._residual = residual

        layers: list[nn.Module] = [
            _conv(dim, in_ch, out_ch),
            _bn(dim, out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers += [
            _conv(dim, out_ch, out_ch),
            _bn(dim, out_ch),
        ]
        # Final ReLU is applied AFTER the residual add (or directly if no shortcut).
        self.body = nn.Sequential(*layers)
        if dropout > 0.0:
            self.dropout: nn.Module | None = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.shortcut: nn.Module | None
        if residual and in_ch != out_ch:
            self.shortcut = _conv(dim, in_ch, out_ch, k=1, padding=0, bias=False)
        else:
            self.shortcut = None

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out: Tensor = cast(Tensor, self.body(x))
        if self._residual:
            identity: Tensor = (
                x if self.shortcut is None else cast(Tensor, self.shortcut(x))
            )
            out = out + identity
        out = F.relu(out)
        if self.dropout is not None:
            out = cast(Tensor, self.dropout(out))
        return out


class _EncoderBlock(nn.Module):
    """Encoder stage: DoubleConv then MaxPool(2)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dim: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch, dim, dropout, residual)
        self.pool = _pool(dim)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        skip: Tensor = self.conv.forward(x)
        pooled: Tensor = cast(Tensor, self.pool(skip))
        return pooled, skip


class _DecoderBlock(nn.Module):
    """Decoder stage: Upsample → pad → cat(skip) → DoubleConv."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        dim: int = 2,
        bilinear: bool = False,
        dropout: float = 0.0,
        residual: bool = False,
    ) -> None:
        super().__init__()
        self._dim = dim
        if bilinear:
            self.up: nn.Module = nn.Upsample(
                scale_factor=2, mode=_interp_mode(dim), align_corners=True
            )
            self.conv = _DoubleConv(in_ch + skip_ch, out_ch, dim, dropout, residual)
        else:
            self.up = _deconv(dim, in_ch, in_ch // 2)
            self.conv = _DoubleConv(
                in_ch // 2 + skip_ch, out_ch, dim, dropout, residual
            )

    @override
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:  # type: ignore[override]
        x_up: Tensor = cast(Tensor, self.up(x))

        # Pad x_up to match skip spatial dims (handles non-power-of-2 inputs).
        if self._dim == 3:
            sD = int(skip.shape[2])
            sH = int(skip.shape[3])
            sW = int(skip.shape[4])
            uD = int(x_up.shape[2])
            uH = int(x_up.shape[3])
            uW = int(x_up.shape[4])
            dD, dH, dW = sD - uD, sH - uH, sW - uW
            if dD or dH or dW:
                # F.pad order: (W_l, W_r, H_t, H_b, D_f, D_b)
                x_up = F.pad(x_up, (0, dW, 0, dH, 0, dD))
        else:
            sH = int(skip.shape[2])
            sW = int(skip.shape[3])
            uH = int(x_up.shape[2])
            uW = int(x_up.shape[3])
            dH, dW = sH - uH, sW - uW
            if dH or dW:
                x_up = F.pad(x_up, (0, dW, 0, dH))

        combined: Tensor = lucid.cat([x_up, skip], dim=1)
        return self.conv.forward(combined)


# ---------------------------------------------------------------------------
# U-Net model
# ---------------------------------------------------------------------------


class UNetForSemanticSegmentation(PretrainedModel):
    r"""U-Net encoder-decoder for semantic segmentation (Ronneberger et al., MICCAI 2015).

    The canonical encoder-decoder segmentation architecture.  A contracting
    path (encoder) halves the spatial resolution and doubles the channel
    count at every stage; an expanding path (decoder) inverts this, with
    **skip connections** concatenating each encoder feature map onto the
    corresponding decoder stage to preserve fine localisation detail that
    would otherwise be lost during downsampling.

    Each stage uses a *DoubleConv* block (two 3x3 convolutions with
    BatchNorm and ReLU); the optional ``block="res"`` ResUNet variant
    adds an identity shortcut inside each DoubleConv to ease gradient
    flow.  Switching ``config.dim`` to 3 swaps every Conv2d / BatchNorm2d
    / MaxPool2d for its 3-D counterpart, yielding the volumetric U-Net
    used in biomedical 3-D segmentation (Cicek et al., 2016).

    Parameters
    ----------
    config : UNetConfig
        Frozen architecture spec.  Use the factories (:func:`unet`,
        :func:`res_unet_2d`, :func:`unet_3d`, :func:`res_unet_3d`) for
        standard variants.

    Attributes
    ----------
    config : UNetConfig
        Stored copy of the config that built this model.
    encoders : list[_EncoderBlock]
        ``config.depth`` encoder stages, each producing one downsampled
        feature and one skip feature.  Channel widths follow
        ``base_channels * 2 ** i`` for i in 0 .. depth - 1.
    bottleneck : _DoubleConv
        DoubleConv at the bottom of the U with ``base_channels * 2 ** depth``
        channels.
    decoders : list[_DecoderBlock]
        ``config.depth`` decoder stages applied bottom-up; each
        upsamples (transpose conv or trilinear/bilinear interpolation
        when ``bilinear=True``), concatenates the matching skip feature,
        and runs another DoubleConv.
    head : nn.Conv2d or nn.Conv3d
        Final 1x1 (or 1x1x1) convolution mapping the topmost decoder
        feature to ``num_classes`` channels.

    Notes
    -----
    See Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015 (arXiv:1505.04597).  The defining
    skip-connection update at decoder stage :math:`s` is

    .. math::

        d_s = \mathrm{DoubleConv}\!\bigl(
                \mathrm{Concat}\bigl(\mathrm{Up}(d_{s+1}),\; e_s\bigr)
              \bigr),

    with :math:`e_s` the encoder feature at the same spatial scale.
    The original paper applies the network at 572x572 input and outputs
    388x388 unpadded; this implementation uses padding throughout so
    the output spatial size matches the input.  ResUNet (Zhang et al.,
    2018) and 3-D U-Net (Cicek et al., 2016) are covered by the
    ``block`` and ``dim`` config switches respectively.

    Examples
    --------
    Standard 2-D U-Net for 2-class biomedical segmentation:

    >>> import lucid
    >>> from lucid.models.vision.unet import unet
    >>> model = unet()
    >>> x = lucid.randn(1, 1, 256, 256)
    >>> out = model(x)
    >>> out.logits.shape   # (B, num_classes, H, W)
    (1, 2, 256, 256)

    3-D U-Net on a volumetric input:

    >>> from lucid.models.vision.unet import unet_3d
    >>> model = unet_3d()
    >>> x = lucid.randn(1, 1, 64, 64, 64)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 2, 64, 64, 64)
    """

    config_class: ClassVar[type[UNetConfig]] = UNetConfig
    base_model_prefix: ClassVar[str] = "unet"

    def __init__(self, config: UNetConfig) -> None:
        super().__init__(config)
        self._cfg = config

        ch = config.base_channels
        depth = config.depth
        dropout = config.dropout
        dim = config.dim
        residual = config.block == "res"

        # Encoder stages
        self.encoders: list[_EncoderBlock] = []
        in_ch = config.in_channels
        enc_channels: list[int] = []
        for i in range(depth):
            out_ch = ch * (2**i)
            block = _EncoderBlock(in_ch, out_ch, dim, dropout, residual)
            self.add_module(f"encoder_{i}", block)
            self.encoders.append(block)
            enc_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = ch * (2**depth)
        self.bottleneck = _DoubleConv(in_ch, bottleneck_ch, dim, dropout, residual)

        # Decoder stages (reverse order)
        self.decoders: list[_DecoderBlock] = []
        dec_in = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_channels[i]
            dec_out = enc_channels[i]
            dec_block = _DecoderBlock(
                dec_in,
                skip_ch,
                dec_out,
                dim=dim,
                bilinear=config.bilinear,
                dropout=dropout,
                residual=residual,
            )
            self.add_module(f"decoder_{i}", dec_block)
            self.decoders.append(dec_block)
            dec_in = dec_out

        # Segmentation head
        self.head = _conv(
            dim, enc_channels[0], config.num_classes, k=1, padding=0, bias=True
        )

    @override
    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: Tensor | None = None,
    ) -> SemanticSegmentationOutput:
        """Run U-Net forward pass."""
        dim = self._cfg.dim
        in_spatial = tuple(int(s) for s in x.shape[2:])

        skips: list[Tensor] = []
        feat = x
        for enc in self.encoders:
            feat, skip = enc.forward(feat)
            skips.append(skip)

        feat = self.bottleneck.forward(feat)

        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            feat = dec.forward(feat, skip)

        logits: Tensor = cast(Tensor, self.head(feat))

        # Ensure output spatial matches input
        out_spatial = tuple(int(s) for s in logits.shape[2:])
        if out_spatial != in_spatial:
            logits = F.interpolate(
                logits,
                size=in_spatial,
                mode=_interp_mode(dim),
                align_corners=False,
            )

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return SemanticSegmentationOutput(logits=logits, loss=loss)
