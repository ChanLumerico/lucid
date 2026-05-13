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
    depth stages of DoubleConv followed by MaxPool2d(2).
    Channels: [base, base×2, base×4, …, base×2^(depth-1)]

  Bottleneck:
    DoubleConv with channels base×2^depth (no pooling).

  Decoder (right side):
    depth stages, each:
      1. Upsample ×2 (bilinear or ConvTranspose2d)
      2. Pad to match skip-connection spatial size if needed
      3. Concatenate with encoder skip feature
      4. DoubleConv → output channels halved

  Head:
    Conv2d(base_channels, num_classes, 1)

  Output: logits (B, num_classes, H, W) — same size as input.

Losses (training)
-----------------
  Requires integer ``targets`` of shape (B, H, W).
  Uses cross-entropy loss: F.cross_entropy(logits, targets).
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models.vision.unet._config import UNetConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _DoubleConv(nn.Module):
    """Two sequential Conv3×3-BN-ReLU blocks with optional dropout.

    Args:
        in_ch:   Input channel count.
        out_ch:  Output channel count.
        dropout: Dropout probability applied after each ReLU (0 = off).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        layers += [
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(p=dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.net(x))


class _EncoderBlock(nn.Module):
    """Encoder stage: DoubleConv then MaxPool2d(2).

    Returns both the pooled output (passed deeper) and the skip feature
    (passed to the corresponding decoder stage).
    """

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv = _DoubleConv(in_ch, out_ch, dropout)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        """Run encoder block.

        Returns:
            Tuple of (pooled, skip) where:
              pooled — (B, out_ch, H/2, W/2) — passed to next stage.
              skip   — (B, out_ch, H, W)    — stored for decoder.
        """
        skip: Tensor = self.conv.forward(x)
        pooled: Tensor = cast(Tensor, self.pool(skip))
        return pooled, skip


class _DecoderBlock(nn.Module):
    """Decoder stage: Upsample → pad → cat(skip) → DoubleConv.

    Args:
        in_ch:    Input channel count (from deeper decoder stage).
        skip_ch:  Skip connection channel count.
        out_ch:   Output channel count.
        bilinear: If True, use bilinear interpolation + Conv2d for upsample;
                  otherwise use ConvTranspose2d.
        dropout:  Dropout probability in DoubleConv.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        bilinear: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if bilinear:
            self.up: nn.Module = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            # After concat: in_ch + skip_ch channels
            self.conv = _DoubleConv(in_ch + skip_ch, out_ch, dropout)
        else:
            # ConvTranspose halves channels and doubles spatial
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
            # After concat: in_ch//2 + skip_ch channels
            self.conv = _DoubleConv(in_ch // 2 + skip_ch, out_ch, dropout)

    def forward(  # type: ignore[override]
        self, x: Tensor, skip: Tensor
    ) -> Tensor:
        """Decode one level.

        Args:
            x:    (B, in_ch, H', W') — decoder tensor from deeper stage.
            skip: (B, skip_ch, 2H', 2W') — encoder skip tensor.

        Returns:
            (B, out_ch, 2H', 2W')
        """
        x_up: Tensor = cast(Tensor, self.up(x))

        # Pad x_up to match skip spatial dimensions if they differ
        # (can happen when input spatial dimensions are not powers of 2)
        skip_H = int(skip.shape[2])
        skip_W = int(skip.shape[3])
        up_H = int(x_up.shape[2])
        up_W = int(x_up.shape[3])
        diff_H = skip_H - up_H
        diff_W = skip_W - up_W
        if diff_H != 0 or diff_W != 0:
            # F.pad(input, (left, right, top, bottom))
            x_up = F.pad(x_up, (0, diff_W, 0, diff_H))

        combined: Tensor = lucid.cat([x_up, skip], dim=1)
        return self.conv.forward(combined)


# ---------------------------------------------------------------------------
# U-Net model
# ---------------------------------------------------------------------------


class UNetForSemanticSegmentation(PretrainedModel):
    """U-Net semantic segmentation model (Ronneberger et al., MICCAI 2015).

    Input contract
    --------------
    ``x``       : (B, in_channels, H, W) image batch.
    ``targets`` : optional (B, H, W) integer segmentation masks for training.

    Output contract
    ---------------
    ``SemanticSegmentationOutput``:
      ``logits`` : (B, num_classes, H, W) — same spatial resolution as input.
      ``loss``   : cross-entropy loss when targets provided.
    """

    config_class: ClassVar[type[UNetConfig]] = UNetConfig
    base_model_prefix: ClassVar[str] = "unet"

    def __init__(self, config: UNetConfig) -> None:
        super().__init__(config)
        self._cfg = config

        ch = config.base_channels
        depth = config.depth
        dropout = config.dropout

        # Encoder stages
        self.encoders: list[_EncoderBlock] = []
        in_ch = config.in_channels
        enc_channels: list[int] = []
        for i in range(depth):
            out_ch = ch * (2 ** i)
            block = _EncoderBlock(in_ch, out_ch, dropout)
            self.add_module(f"encoder_{i}", block)
            self.encoders.append(block)
            enc_channels.append(out_ch)
            in_ch = out_ch

        # Bottleneck
        bottleneck_ch = ch * (2 ** depth)
        self.bottleneck = _DoubleConv(in_ch, bottleneck_ch, dropout)

        # Decoder stages (reverse order)
        self.decoders: list[_DecoderBlock] = []
        dec_in = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            skip_ch = enc_channels[i]
            dec_out = enc_channels[i]
            dec_block = _DecoderBlock(
                dec_in, skip_ch, dec_out,
                bilinear=config.bilinear,
                dropout=dropout,
            )
            self.add_module(f"decoder_{i}", dec_block)
            self.decoders.append(dec_block)
            dec_in = dec_out

        # Segmentation head
        self.head = nn.Conv2d(enc_channels[0], config.num_classes, 1)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: Tensor | None = None,
    ) -> SemanticSegmentationOutput:
        """Run U-Net forward pass.

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

        # Decoder path (skips in reverse order)
        for i, dec in enumerate(self.decoders):
            skip = skips[-(i + 1)]
            feat = dec.forward(feat, skip)

        # Segmentation head
        logits: Tensor = cast(Tensor, self.head(feat))

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
