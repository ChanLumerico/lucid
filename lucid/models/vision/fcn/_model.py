"""FCN — Fully Convolutional Network (Long et al., CVPR 2015).

Paper: "Fully Convolutional Networks for Semantic Segmentation"

Key innovation
--------------
Replace all fully-connected layers in a classification CNN with
convolutions so the network can accept arbitrary input sizes and produce
a dense per-pixel prediction.  Skip connections from shallower stages
fuse coarse semantic information with fine spatial detail.

Architecture (ResNet variant)
------------------------------
  ResNet backbone with dilated convolutions:
    - layer3 uses dilation=2 (stride replaced) → 1/8 spatial resolution
    - layer4 uses dilation=4 (stride replaced) → still 1/8

  FCN head (on layer4 output, 2048 channels):
    Conv2d(2048, 512, 3, pad=1) → BN → ReLU → Dropout → Conv2d(512, K, 1)
    Bilinear upsample ×8 → (B, K, H, W)

  Auxiliary head (on layer3 output, 1024 channels, training only):
    Conv2d(1024, 256, 3, pad=1) → BN → ReLU → Dropout → Conv2d(256, K, 1)
    Bilinear upsample ×8 → (B, K, H, W)

Losses (training)
-----------------
  primary loss:   cross-entropy on main head output vs. targets
  auxiliary loss: cross-entropy on aux head output vs. targets
  total loss:     primary + 0.4 × auxiliary
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._output import SemanticSegmentationOutput
from lucid.models.vision.fcn._config import FCNConfig


# ---------------------------------------------------------------------------
# ResNet backbone (dilated, self-contained)
# ---------------------------------------------------------------------------


class _Bottleneck(nn.Module):
    """1×1 → 3×3 (dilated) → 1×1 bottleneck block."""

    expansion: ClassVar[int] = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        stride: int = 1,
        dilation: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(
            mid_ch,
            mid_ch,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x
        out: Tensor = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        out = F.relu(cast(Tensor, self.bn2(cast(Tensor, self.conv2(out)))))
        out = cast(Tensor, self.bn3(cast(Tensor, self.conv3(out))))
        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))
        return F.relu(out + identity)


def _make_layer(
    in_ch: int,
    mid_ch: int,
    num_blocks: int,
    stride: int = 1,
    dilation: int = 1,
) -> tuple[nn.Sequential, int]:
    """Build one ResNet stage with optional dilation."""
    out_ch = mid_ch * 4
    ds: nn.Module | None = None
    if stride != 1 or in_ch != out_ch:
        ds = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
    blocks: list[nn.Module] = [
        _Bottleneck(in_ch, mid_ch, stride=stride, dilation=dilation, downsample=ds)
    ]
    for _ in range(1, num_blocks):
        blocks.append(_Bottleneck(out_ch, mid_ch, dilation=dilation))
    return nn.Sequential(*blocks), out_ch


class _DilatedResNet(nn.Module):
    """ResNet backbone with dilated convolutions for dense prediction.

    Replaces stride in layer3 and layer4 with dilation so the output
    feature map retains 1/8 the input spatial resolution rather than 1/32.

    Args:
        in_channels: Number of input image channels.
        layers:      Per-stage repetition counts, e.g. (3, 4, 6, 3).

    Attributes:
        c4_channels: Channel count of layer3 output (for auxiliary head).
        c5_channels: Channel count of layer4 output (for FCN head).
    """

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, int, int, int],
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1, c2 = _make_layer(64, 64, layers[0], stride=1)
        self.layer2, c3 = _make_layer(c2, 128, layers[1], stride=2)
        # layer3: stride=1, dilation=2 → still 1/8 resolution
        self.layer3, c4 = _make_layer(c3, 256, layers[2], stride=1, dilation=2)
        # layer4: stride=1, dilation=4 → still 1/8 resolution
        self.layer4, c5 = _make_layer(c4, 512, layers[3], stride=1, dilation=4)

        self.c4_channels: int = c4
        self.c5_channels: int = c5

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Return (c4, c5) feature maps.

        Returns:
            c4: (B, c4_channels, H/8, W/8) — layer3 output.
            c5: (B, c5_channels, H/8, W/8) — layer4 output.
        """
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.maxpool(x))
        x = cast(Tensor, self.layer1(x))
        x = cast(Tensor, self.layer2(x))
        c4: Tensor = cast(Tensor, self.layer3(x))
        c5: Tensor = cast(Tensor, self.layer4(c4))
        return c4, c5


# ---------------------------------------------------------------------------
# FCN segmentation head
# ---------------------------------------------------------------------------


class _FCNHead(nn.Sequential):
    """FCN segmentation head: Conv3×3-BN-ReLU-Dropout-Conv1×1.

    Args:
        in_channels:  Input feature channel count.
        hidden_channels: Intermediate channel count (default 512 for main head).
        num_classes:  Number of output segmentation classes.
        dropout:      Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Conv2d(hidden_channels, num_classes, 1),
        )


# ---------------------------------------------------------------------------
# FCN model
# ---------------------------------------------------------------------------


class FCNForSemanticSegmentation(PretrainedModel):
    """FCN semantic segmentation model (Long et al., CVPR 2015).

    Input contract
    --------------
    ``x``       : (B, in_channels, H, W) image batch.
    ``targets`` : optional (B, H, W) integer segmentation masks for training.

    Output contract
    ---------------
    ``SemanticSegmentationOutput``:
      ``logits`` : (B, num_classes, H, W) — same spatial resolution as input.
      ``loss``   : primary cross-entropy + 0.4 × auxiliary cross-entropy
                   when targets are provided.
    """

    config_class: ClassVar[type[FCNConfig]] = FCNConfig
    base_model_prefix: ClassVar[str] = "fcn"

    def __init__(self, config: FCNConfig) -> None:
        super().__init__(config)
        self._cfg = config

        # Backbone layer counts by name
        _layer_counts: dict[str, tuple[int, int, int, int]] = {
            "resnet50":  (3, 4, 6, 3),
            "resnet101": (3, 4, 23, 3),
        }
        layers = _layer_counts.get(config.backbone, (3, 4, 6, 3))

        self.backbone = _DilatedResNet(config.in_channels, layers)

        self.classifier = _FCNHead(
            self.backbone.c5_channels,
            config.classifier_hidden_channels,
            config.num_classes,
            config.dropout,
        )
        self.aux_classifier = _FCNHead(
            self.backbone.c4_channels,
            config.aux_hidden_channels,
            config.num_classes,
            config.dropout,
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        targets: Tensor | None = None,
    ) -> SemanticSegmentationOutput:
        """Run FCN forward pass.

        Args:
            x:       (B, C, H, W) image batch.
            targets: Optional (B, H, W) integer ground-truth masks.

        Returns:
            ``SemanticSegmentationOutput`` with logits (B, num_classes, H, W)
            and optional cross-entropy loss.
        """
        iH = int(x.shape[2])
        iW = int(x.shape[3])

        # Backbone: (B, c4, H/8, W/8) and (B, c5, H/8, W/8)
        c4, c5 = self.backbone.forward(x)

        # Main head
        main_feat: Tensor = cast(Tensor, self.classifier(c5))
        logits: Tensor = F.interpolate(
            main_feat, size=(iH, iW), mode="bilinear", align_corners=False
        )

        loss: Tensor | None = None
        if targets is not None:
            main_loss: Tensor = F.cross_entropy(logits, targets)

            # Auxiliary head — only computed during training
            aux_feat: Tensor = cast(Tensor, self.aux_classifier(c4))
            aux_logits: Tensor = F.interpolate(
                aux_feat, size=(iH, iW), mode="bilinear", align_corners=False
            )
            aux_loss: Tensor = F.cross_entropy(aux_logits, targets)
            loss = main_loss + 0.4 * aux_loss

        return SemanticSegmentationOutput(logits=logits, loss=loss)
