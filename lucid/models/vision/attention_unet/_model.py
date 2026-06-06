"""Attention U-Net (Oktay et al., MIDL 2018).

Paper: "Attention U-Net: Learning Where to Look for the Pancreas"

Key innovation
--------------
Each skip connection in the U-Net decoder is gated by a soft spatial
attention map.  The attention gate (AG) takes two inputs:

* **x**: (B, F, H, W) — the skip feature from the encoder.
* **g**: (B, G, H', W') — the gating signal from the decoder (coarser).

Steps inside the AG:
  1. Wx = Conv1x1(F → inter) applied to x,   then upsample to (H, W).
  2. Wg = Conv1x1(G → inter) applied to g.
  3. psi = sigmoid(Conv1x1(inter → 1)(ReLU(Wx + Wg))) — spatial map.
  4. Output = x * psi   (element-wise, broadcast over channels).

Architecture
------------
  Encoder: depth stages of DoubleConv + MaxPool2d (halves spatial dims).
  Bottleneck: DoubleConv at the deepest level.
  Decoder: depth stages of Upsample/ConvTranspose2d
             → AttentionGate on skip
             → Cat(gated_skip, upsampled_decoder)
             → DoubleConv.
  Head: Conv2d(base_channels, num_classes, 1).

Losses (training)
-----------------
  Requires integer ``targets`` of shape (B, H, W).
  Uses cross-entropy loss: F.cross_entropy(logits, targets).
"""

from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models.vision.attention_unet._config import AttentionUNetConfig

# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Module):
    """Two sequential Conv3x3-BN-ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.net(x))


@final
class _AttentionGate(nn.Module):
    """Soft attention gate for U-Net skip connections.

    Args:
        x_channels: Channels in the skip feature x (F).
        g_channels: Channels in the gating signal g (G).
        inter_channels: Intermediate projection size.
    """

    def __init__(self, x_channels: int, g_channels: int, inter_channels: int) -> None:
        super().__init__()
        self.Wx = nn.Conv2d(x_channels, inter_channels, 1, bias=True)
        self.Wg = nn.Conv2d(g_channels, inter_channels, 1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    @override
    def forward(self, x: Tensor, g: Tensor) -> Tensor:  # type: ignore[override]
        """Apply attention gate.

        Args:
            x: (B, F, H, W) — skip features (finer scale).
            g: (B, G, H/2, W/2) — gating signal (coarser).

        Returns:
            (B, F, H, W) — gated skip features.
        """
        # 1. Project skip: (B, inter, H, W)
        wx: Tensor = cast(Tensor, self.Wx(x))
        # 2. Project gate and upsample: (B, inter, H', W') → (B, inter, H, W)
        wg_raw: Tensor = cast(Tensor, self.Wg(g))
        wg: Tensor = cast(Tensor, self.up(wg_raw))
        # 3. Combine and compute attention map
        combined: Tensor = F.relu(wx + wg)
        att: Tensor = F.sigmoid(cast(Tensor, self.psi(combined)))  # (B, 1, H, W)
        return x * att


class _EncoderBlock(nn.Module):
    """Encoder stage: DoubleConv → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2, stride=2)

    @override
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        """Returns (pooled, skip)."""
        skip: Tensor = self.conv.forward(x)
        pooled: Tensor = cast(Tensor, self.pool(skip))
        return pooled, skip


class _DecoderBlock(nn.Module):
    """Decoder stage: Upsample → AttentionGate → Cat → DoubleConv."""

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = False,
    ) -> None:
        super().__init__()
        inter_ch = out_ch // 2 if out_ch // 2 > 0 else 1
        self.gate = _AttentionGate(skip_ch, in_ch, inter_ch)
        if bilinear:
            self.up: nn.Module = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = _DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    @override
    def forward(self, x: Tensor, skip: Tensor) -> Tensor:  # type: ignore[override]
        """Decode one level.

        Args:
            x:    (B, in_ch, H', W') — decoder tensor (coarser scale).
            skip: (B, skip_ch, 2H', 2W') — encoder skip tensor.

        Returns:
            (B, out_ch, 2H', 2W')
        """
        x_up: Tensor = cast(Tensor, self.up(x))
        gated_skip: Tensor = self.gate.forward(skip, x)
        combined: Tensor = lucid.cat([x_up, gated_skip], dim=1)
        return self.conv.forward(combined)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AttentionUNetForSemanticSegmentation(PretrainedModel):
    r"""Attention U-Net for medical image segmentation (Oktay et al., MIDL 2018).

    A U-Net variant that inserts a soft **attention gate** on every skip
    connection.  Each gate fuses the encoder feature :math:`x^\ell` (the
    "input" branch) with the up-sampled decoder feature :math:`g^\ell`
    (the "gating signal") via

    .. math::

        \alpha^\ell = \sigma\!\bigl(\psi^\top \tanh(W_x x^\ell + W_g g^\ell)\bigr),
        \qquad
        \hat{x}^\ell = \alpha^\ell \odot x^\ell,

    suppressing skip-feature activations outside the regions of interest
    highlighted by the decoder.  This focuses the decoder's attention on
    relevant anatomy and removes the "noise" carried over from encoder
    layers — a consistent +1-3 Dice gain on medical-imaging benchmarks
    in the original paper.

    Parameters
    ----------
    config : AttentionUNetConfig
        Frozen architecture spec.  Use :func:`attention_unet` for the
        standard 4-level configuration.

    Attributes
    ----------
    config : AttentionUNetConfig
        Stored copy of the config that built this model.
    encoders : list[_EncoderBlock]
        ``config.depth`` 2-D encoder stages (DoubleConv + MaxPool).
    bottleneck : _DoubleConv
        DoubleConv at the bottom of the U.
    decoders : list[_DecoderBlock]
        ``config.depth`` decoder stages, each applying an attention gate
        to the corresponding skip feature *before* concatenating it into
        the decoder DoubleConv.
    head : nn.Conv2d
        1x1 convolution producing ``num_classes`` channels.

    Notes
    -----
    See Oktay et al., "Attention U-Net: Learning Where to Look for the
    Pancreas", MIDL 2018 (arXiv:1804.03999).  The additive attention
    gate is identical to that of Bahdanau et al. (2015) generalised to
    feature maps:

    .. math::

        q^\ell = \psi^\top \sigma_1\!\bigl(W_x x^\ell + W_g g^\ell + b_g\bigr),
        \quad
        \alpha^\ell = \sigma_2(q^\ell + b_\psi),

    with :math:`\sigma_1 = \mathrm{ReLU}`, :math:`\sigma_2 = \mathrm{sigmoid}`.
    The intermediate channel count is typically half the input channels.

    Examples
    --------
    >>> import lucid
    >>> from lucid.models.vision.attention_unet import attention_unet
    >>> model = attention_unet()
    >>> x = lucid.randn(1, 1, 256, 256)
    >>> out = model(x)
    >>> out.logits.shape   # (B, num_classes, H, W)
    (1, 2, 256, 256)
    """

    config_class: ClassVar[type[AttentionUNetConfig]] = AttentionUNetConfig
    base_model_prefix: ClassVar[str] = "attention_unet"

    def __init__(self, config: AttentionUNetConfig) -> None:
        super().__init__(config)
        self._cfg = config

        ch = config.base_channels
        depth = config.depth

        # Encoder stages
        self.encoders: list[_EncoderBlock] = []
        in_ch = config.in_channels
        enc_channels: list[int] = []  # output channels at each stage
        for i in range(depth):
            out_ch = ch * (2**i)
            block = _EncoderBlock(in_ch, out_ch)
            self.add_module(f"encoder_{i}", block)
            self.encoders.append(block)
            enc_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = ch * (2**depth)
        self.bottleneck = _DoubleConv(in_ch, bottleneck_ch)

        # Decoder stages (reverse order)
        self.decoders: list[_DecoderBlock] = []
        dec_in = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_channels[i]
            dec_out = enc_channels[i]
            dec_block = _DecoderBlock(
                dec_in, skip_ch, dec_out, bilinear=config.bilinear
            )
            self.add_module(f"decoder_{i}", dec_block)
            self.decoders.append(dec_block)
            dec_in = dec_out

        # Segmentation head
        self.head = nn.Conv2d(enc_channels[0], config.num_classes, 1)

    @override
    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: Tensor | None = None,
    ) -> SemanticSegmentationOutput:
        """Run Attention U-Net.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional (B, H, W) integer ground-truth masks.

        Returns:
            ``SemanticSegmentationOutput`` with logits (B, num_classes, H, W)
            and optional cross-entropy loss.
        """
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # Encoder path
        skips: list[Tensor] = []
        feat = x
        for enc in self.encoders:
            feat, skip = enc.forward(feat)
            skips.append(skip)

        # Bottleneck
        feat = self.bottleneck.forward(feat)

        # Decoder path
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            feat = dec.forward(feat, skip)

        # Segmentation head
        logits: Tensor = cast(Tensor, self.head(feat))  # (B, num_classes, H, W)

        # Ensure output matches input spatial size
        out_H = int(logits.shape[2])
        out_W = int(logits.shape[3])
        if out_H != iH or out_W != iW:
            logits = F.interpolate(
                logits, size=(iH, iW), mode="bilinear", align_corners=False
            )

        loss: Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return SemanticSegmentationOutput(logits=logits, loss=loss)
