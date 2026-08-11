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

Divergence, deliberate: 2-D
---------------------------
  The paper's Implementation Details state "we propose a 3D-model to
  capture sufficient semantic context", and every released network is
  Conv3d / BatchNorm3d with trilinear resampling — it was built for CT
  volumes.  This is a 2-D model: the attention gate, the encoder and the
  decoder are all Conv2d.  The gating mechanism, which is the paper's
  actual contribution, is unchanged; only the convolution rank differs.
  A caller with volumetric data wants a 3-D port, not this module, and
  no 3-D checkpoint will load into it.
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
        # Section 3.2: "input feature-maps are downsampled to the resolution
        # of gating signal".  Wx therefore carries stride 2, so the whole
        # ReLU / psi / sigmoid stack runs on the coarse grid -- a quarter of
        # the activations, and the attention coefficients are decided at the
        # scale the gate actually carries semantics at.
        self.Wx = nn.Conv2d(x_channels, inter_channels, 1, stride=2, bias=True)
        self.Wg = nn.Conv2d(g_channels, inter_channels, 1, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, 1, bias=True)

    @override
    def forward(self, x: Tensor, g: Tensor) -> Tensor:  # type: ignore[override]
        """Apply attention gate.

        Args:
            x: (B, F, H, W) — skip features (finer scale).
            g: (B, G, H/2, W/2) — gating signal (coarser).

        Returns:
            (B, F, H, W) — gated skip features.  The coefficients themselves
            are computed at the gating resolution and resampled back up.
        """
        # 1. Project *and* downsample the skip: (B, inter, H/2, W/2)
        wx: Tensor = cast(Tensor, self.Wx(x))
        # 2. Project gate and resample to wx's *actual* size.  A fixed
        # scale_factor only lands on the right shape when every spatial
        # dimension is even all the way down: MaxPool2d floors, so a 25x25
        # skip pools to 12x12 and the two grids disagree.  The reference
        # resamples to ``theta_x_size[2:]`` for exactly this reason.
        wg_raw: Tensor = cast(Tensor, self.Wg(g))
        wg: Tensor = F.interpolate(
            wg_raw,
            size=(int(wx.shape[2]), int(wx.shape[3])),
            mode="bilinear",
            align_corners=True,
        )
        # 3. Combine and compute the attention map on the gating grid
        combined: Tensor = F.relu(wx + wg)
        att: Tensor = F.sigmoid(cast(Tensor, self.psi(combined)))
        # 4. "Grid resampling of attention coefficients" back to x's size
        att = F.interpolate(
            att,
            size=(int(x.shape[2]), int(x.shape[3])),
            mode="bilinear",
            align_corners=True,
        )
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
    """Decoder stage: Upsample → AttentionGate → Cat → DoubleConv.

    ``gated=False`` (the finest level) skips the gate entirely.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = False,
        gated: bool = True,
    ) -> None:
        super().__init__()
        inter_ch = out_ch // 2 if out_ch // 2 > 0 else 1
        # Section 3.2: "low-level feature-maps, i.e. the first skip
        # connections, are not used in the gating function since they do not
        # represent the input data in a high dimensional space."  The
        # reference builds 3 gates for a 4-level U-Net, passing conv1
        # straight through.
        self.gate: _AttentionGate | None = (
            _AttentionGate(skip_ch, in_ch, inter_ch) if gated else None
        )
        if bilinear:
            self.up: nn.Module = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = _DoubleConv(in_ch + skip_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch)

    @override
    def forward(  # type: ignore[override]
        self, x: Tensor, skip: Tensor, gate: Tensor | None = None
    ) -> Tensor:
        """Decode one level.

        Args:
            x:    (B, in_ch, H', W') — decoder tensor (coarser scale).
            skip: (B, skip_ch, 2H', 2W') — encoder skip tensor.

        Returns:
            (B, out_ch, 2H', 2W')
        """
        x_up: Tensor = cast(Tensor, self.up(x))
        # The deepest stage is handed the projected gating signal; shallower
        # stages gate on the previous decoder output, as the reference does.
        g_src = x if gate is None else gate
        gated_skip: Tensor = (
            skip if self.gate is None else self.gate.forward(skip, g_src)
        )
        # The decoder's own upsample is also a fixed factor of 2, so pad it
        # up to the skip whenever the encoder's floor-division lost a row or
        # column.  Same guard the sibling U-Net applies.
        sH, sW = int(gated_skip.shape[2]), int(gated_skip.shape[3])
        uH, uW = int(x_up.shape[2]), int(x_up.shape[3])
        if sH != uH or sW != uW:
            x_up = F.pad(x_up, (0, sW - uW, 0, sH - uH))
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

        \alpha^\ell = \sigma\!\bigl(\psi^\top \mathrm{ReLU}(W_x x^\ell + W_g g^\ell)\bigr),
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
        # The reference does not hand the raw bottleneck to the gates: a
        # ``UnetGridGatingSignal`` (1x1 conv + BN + ReLU) projects it first,
        # so the gating signal is a learned summary rather than whatever the
        # deepest convolution happened to leave behind.
        self.gating = nn.Sequential(
            nn.Conv2d(bottleneck_ch, bottleneck_ch, 1, bias=False),
            nn.BatchNorm2d(bottleneck_ch),
            nn.ReLU(inplace=True),
        )

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
                bilinear=config.bilinear,
                gated=i > 0,
            )
            self.add_module(f"decoder_{i}", dec_block)
            self.decoders.append(dec_block)
            dec_in = dec_out

        # Segmentation head.
        #
        # Deep supervision is NOT implemented.  The paper's own model
        # (``unet_CT_multi_att_dsv``) attaches a per-level 1x1 conv, upsamples
        # each to the input resolution and sums the losses; the "dsv" in its
        # name is that mechanism.  It is a training-time signal only — the
        # inference path is identical either way — so its absence changes no
        # forward result, but a from-scratch run will not reproduce the
        # paper's convergence without it.
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
        gating: Tensor = cast(Tensor, self.gating(feat))

        # Decoder path
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            feat = dec.forward(feat, skip, gating if i == 0 else None)

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
