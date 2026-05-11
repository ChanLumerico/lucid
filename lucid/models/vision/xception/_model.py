"""Xception backbone and classifier (Chollet, 2017).

Paper: "Xception: Deep Learning with Depthwise Separable Convolutions"

Architecture overview (299×299 input):
    Entry flow:
        Conv 3×3-s2 (32) → Conv 3×3 (64) → ReLU
        Block1: 2× SepConv(128) + skip → MaxPool-s2   (147→74)
        Block2: 2× SepConv(256) + skip → MaxPool-s2   (74→37)
        Block3: 2× SepConv(728) + skip → MaxPool-s2   (37→18? -- see note)
    Middle flow (8×):
        3× SepConv(728) + residual
    Exit flow:
        SepConv(728) + SepConv(1024) + skip → MaxPool-s2
        SepConv(1536) → SepConv(2048)
        AdaptiveAvgPool(1×1) → Dropout → FC

SepConv = Depthwise Conv2d → Pointwise Conv2d (1×1), each followed by BN+ReLU.
"""

from dataclasses import dataclass
from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput
from lucid.models.vision.xception._config import XceptionConfig

# ---------------------------------------------------------------------------
# SepConv: Depthwise + Pointwise Conv, BN, ReLU
# ---------------------------------------------------------------------------


class _SepConv(nn.Module):
    """Depthwise separable convolution: depthwise → pointwise, BN, ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        activate_first: bool = True,
    ) -> None:
        super().__init__()
        self.activate_first = activate_first
        # Depthwise conv (groups=in_channels)
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        # Pointwise conv (1×1)
        self.pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self.activate_first:
            x = F.relu(x)
        x = cast(Tensor, self.dw(x))
        x = cast(Tensor, self.pw(x))
        return cast(Tensor, self.bn(x))


# ---------------------------------------------------------------------------
# Entry flow block
# ---------------------------------------------------------------------------


class _EntryFlowBlock(nn.Module):
    """Entry flow block: 2× SepConv + residual skip + MaxPool stride=2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        activate_first: bool = True,
    ) -> None:
        super().__init__()
        self.sep1 = _SepConv(in_channels, out_channels, activate_first=activate_first)
        self.sep2 = _SepConv(out_channels, out_channels)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        # Residual 1×1 projection (matches spatial stride and channel dim)
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        residual = cast(Tensor, self.skip(x))
        x = cast(Tensor, self.sep1(x))
        x = cast(Tensor, self.sep2(x))
        x = cast(Tensor, self.pool(x))
        return x + residual


# ---------------------------------------------------------------------------
# Middle flow block
# ---------------------------------------------------------------------------


class _MiddleFlowBlock(nn.Module):
    """Middle flow block: 3× SepConv(728) + residual."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.sep1 = _SepConv(channels, channels)
        self.sep2 = _SepConv(channels, channels)
        self.sep3 = _SepConv(channels, channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        residual = x
        x = cast(Tensor, self.sep1(x))
        x = cast(Tensor, self.sep2(x))
        x = cast(Tensor, self.sep3(x))
        return x + residual


# ---------------------------------------------------------------------------
# Exit flow block
# ---------------------------------------------------------------------------


class _ExitFlowBlock(nn.Module):
    """Exit flow: SepConv(728) + SepConv(1024) + residual skip → MaxPool stride=2."""

    def __init__(self) -> None:
        super().__init__()
        self.sep1 = _SepConv(728, 728)
        self.sep2 = _SepConv(728, 1024)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.skip = nn.Sequential(
            nn.Conv2d(728, 1024, 1, stride=2, bias=False),
            nn.BatchNorm2d(1024),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        residual = cast(Tensor, self.skip(x))
        x = cast(Tensor, self.sep1(x))
        x = cast(Tensor, self.sep2(x))
        x = cast(Tensor, self.pool(x))
        return x + residual


# ---------------------------------------------------------------------------
# Xception output dataclass
# ---------------------------------------------------------------------------


@dataclass
class XceptionOutput:
    """Xception classification output."""

    logits: Tensor
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# Xception backbone (task="base")
# ---------------------------------------------------------------------------


class Xception(PretrainedModel, BackboneMixin):
    """Xception feature extractor — outputs (B, 2048, 1, 1) for 299×299 inputs."""

    config_class: ClassVar[type[XceptionConfig]] = XceptionConfig
    base_model_prefix: ClassVar[str] = "xception"

    def __init__(self, config: XceptionConfig) -> None:
        super().__init__(config)
        ic = config.in_channels

        # Entry flow stem: Conv 3×3-s2 (32) → Conv 3×3 (64)
        self.stem_conv1 = nn.Conv2d(ic, 32, 3, stride=2, padding=1, bias=False)
        self.stem_bn1 = nn.BatchNorm2d(32)
        self.stem_conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.stem_bn2 = nn.BatchNorm2d(64)

        # Entry flow blocks (3 blocks)
        self.entry_block1 = _EntryFlowBlock(64, 128, activate_first=False)
        self.entry_block2 = _EntryFlowBlock(128, 256)
        self.entry_block3 = _EntryFlowBlock(256, 728)

        # Middle flow (8 blocks)
        self.middle_block0 = _MiddleFlowBlock(728)
        self.middle_block1 = _MiddleFlowBlock(728)
        self.middle_block2 = _MiddleFlowBlock(728)
        self.middle_block3 = _MiddleFlowBlock(728)
        self.middle_block4 = _MiddleFlowBlock(728)
        self.middle_block5 = _MiddleFlowBlock(728)
        self.middle_block6 = _MiddleFlowBlock(728)
        self.middle_block7 = _MiddleFlowBlock(728)

        # Exit flow
        self.exit_block = _ExitFlowBlock()
        self.exit_sep1 = _SepConv(1024, 1536, activate_first=False)
        self.exit_sep2 = _SepConv(1536, 2048, activate_first=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=128, reduction=4),
            FeatureInfo(stage=2, num_channels=256, reduction=8),
            FeatureInfo(stage=3, num_channels=728, reduction=16),
            FeatureInfo(stage=4, num_channels=2048, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        # Stem
        x = F.relu(cast(Tensor, self.stem_bn1(cast(Tensor, self.stem_conv1(x)))))
        x = F.relu(cast(Tensor, self.stem_bn2(cast(Tensor, self.stem_conv2(x)))))

        # Entry flow
        x = cast(Tensor, self.entry_block1(x))
        x = cast(Tensor, self.entry_block2(x))
        x = cast(Tensor, self.entry_block3(x))

        # Middle flow
        x = cast(Tensor, self.middle_block0(x))
        x = cast(Tensor, self.middle_block1(x))
        x = cast(Tensor, self.middle_block2(x))
        x = cast(Tensor, self.middle_block3(x))
        x = cast(Tensor, self.middle_block4(x))
        x = cast(Tensor, self.middle_block5(x))
        x = cast(Tensor, self.middle_block6(x))
        x = cast(Tensor, self.middle_block7(x))

        # Exit flow
        x = cast(Tensor, self.exit_block(x))
        x = F.relu(cast(Tensor, self.exit_sep1(x)))
        x = F.relu(cast(Tensor, self.exit_sep2(x)))

        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# Xception for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class XceptionForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Xception with Dropout + FC classification head."""

    config_class: ClassVar[type[XceptionConfig]] = XceptionConfig
    base_model_prefix: ClassVar[str] = "xception"

    def __init__(self, config: XceptionConfig) -> None:
        super().__init__(config)
        ic = config.in_channels

        # Entry flow stem
        self.stem_conv1 = nn.Conv2d(ic, 32, 3, stride=2, padding=1, bias=False)
        self.stem_bn1 = nn.BatchNorm2d(32)
        self.stem_conv2 = nn.Conv2d(32, 64, 3, padding=1, bias=False)
        self.stem_bn2 = nn.BatchNorm2d(64)

        # Entry flow blocks
        self.entry_block1 = _EntryFlowBlock(64, 128, activate_first=False)
        self.entry_block2 = _EntryFlowBlock(128, 256)
        self.entry_block3 = _EntryFlowBlock(256, 728)

        # Middle flow
        self.middle_block0 = _MiddleFlowBlock(728)
        self.middle_block1 = _MiddleFlowBlock(728)
        self.middle_block2 = _MiddleFlowBlock(728)
        self.middle_block3 = _MiddleFlowBlock(728)
        self.middle_block4 = _MiddleFlowBlock(728)
        self.middle_block5 = _MiddleFlowBlock(728)
        self.middle_block6 = _MiddleFlowBlock(728)
        self.middle_block7 = _MiddleFlowBlock(728)

        # Exit flow
        self.exit_block = _ExitFlowBlock()
        self.exit_sep1 = _SepConv(1024, 1536, activate_first=False)
        self.exit_sep2 = _SepConv(1536, 2048, activate_first=False)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(2048, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> XceptionOutput:
        # Stem
        x = F.relu(cast(Tensor, self.stem_bn1(cast(Tensor, self.stem_conv1(x)))))
        x = F.relu(cast(Tensor, self.stem_bn2(cast(Tensor, self.stem_conv2(x)))))

        # Entry flow
        x = cast(Tensor, self.entry_block1(x))
        x = cast(Tensor, self.entry_block2(x))
        x = cast(Tensor, self.entry_block3(x))

        # Middle flow
        x = cast(Tensor, self.middle_block0(x))
        x = cast(Tensor, self.middle_block1(x))
        x = cast(Tensor, self.middle_block2(x))
        x = cast(Tensor, self.middle_block3(x))
        x = cast(Tensor, self.middle_block4(x))
        x = cast(Tensor, self.middle_block5(x))
        x = cast(Tensor, self.middle_block6(x))
        x = cast(Tensor, self.middle_block7(x))

        # Exit flow
        x = cast(Tensor, self.exit_block(x))
        x = F.relu(cast(Tensor, self.exit_sep1(x)))
        x = F.relu(cast(Tensor, self.exit_sep2(x)))

        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return XceptionOutput(logits=logits, loss=loss)
