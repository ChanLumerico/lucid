"""Inception-ResNet v2 backbone and classifier (Szegedy et al., 2016).

Paper: "Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning"

Architecture overview (299×299 input):
    Stem (shared with Inception-v4): 299×299 → 35×35 × 384
    Block-A × 5   (residual Inception-A): 35×35 × 384
    Reduction-A   (shared with v4):       17×17 × 1024
    Block-B × 10  (residual Inception-B): 17×17 × 1152
    Reduction-B:                           8×8  × 2016
    Block-C × 5   (residual Inception-C):  8×8  × 2016
    Head: AdaptiveAvgPool(1×1) → Dropout(0.2) → FC(2016, num_classes)

Each residual block applies: output = relu(x + scale * branch_output)
"""

from dataclasses import dataclass
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput
from lucid.models.vision.inception_resnet._config import InceptionResNetConfig

# ---------------------------------------------------------------------------
# Shared Conv-BN-ReLU helper
# ---------------------------------------------------------------------------


class _ConvBnReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        *,
        stride: int = 1,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.relu(cast(Tensor, self.bn(cast(Tensor, self.conv(x)))))


class _ConvBn(nn.Module):
    """Conv2d → BatchNorm2d (no ReLU — used for residual projection)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.bn(cast(Tensor, self.conv(x))))


# ---------------------------------------------------------------------------
# Inception-v4 stem (shared)
# ---------------------------------------------------------------------------


def _build_stem(in_channels: int) -> nn.Sequential:
    """Inception-v4 / Inception-ResNet-v2 stem: 299×299 → 35×35 × 384."""
    return nn.Sequential(
        _ConvBnReLU(in_channels, 32, 3, stride=2),
        _ConvBnReLU(32, 32, 3),
        _ConvBnReLU(32, 64, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        _ConvBnReLU(64, 80, 1),
        _ConvBnReLU(80, 192, 3),
        _ConvBnReLU(192, 384, 3, stride=2),
    )


# ---------------------------------------------------------------------------
# Reduction-A (35×35 → 17×17, shared with Inception-v4)
# ---------------------------------------------------------------------------


class _ReductionA(nn.Module):
    """Reduction-A: 35×35 × 384 → 17×17 × 1024."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch1 = _ConvBnReLU(in_channels, 384, 3, stride=2)
        self.branch2_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch2_b = _ConvBnReLU(192, 224, 3, padding=1)
        self.branch2_c = _ConvBnReLU(224, 256, 3, stride=2)
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(
            Tensor,
            self.branch2_c(
                cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
            ),
        )
        b3 = cast(Tensor, self.branch3(x))
        return lucid.cat([b1, b2, b3], dim=1)


# ---------------------------------------------------------------------------
# Block-A: Residual Inception-A (5× repeated, 35×35 × 384)
# ---------------------------------------------------------------------------


class _BlockA(nn.Module):
    """Residual Inception-A: 35×35 × 384 → 35×35 × 384."""

    def __init__(self, in_channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        # branch1: 1×1 (32)
        self.branch1 = _ConvBnReLU(in_channels, 32, 1)
        # branch2: 1×1(32) → 3×3(32)
        self.branch2_a = _ConvBnReLU(in_channels, 32, 1)
        self.branch2_b = _ConvBnReLU(32, 32, 3, padding=1)
        # branch3: 1×1(32) → 3×3(48) → 3×3(64)
        self.branch3_a = _ConvBnReLU(in_channels, 32, 1)
        self.branch3_b = _ConvBnReLU(32, 48, 3, padding=1)
        self.branch3_c = _ConvBnReLU(48, 64, 3, padding=1)
        # 1×1 projection: 32+32+64=128 → 384
        self.proj = _ConvBn(128, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
        b3 = cast(
            Tensor,
            self.branch3_c(
                cast(Tensor, self.branch3_b(cast(Tensor, self.branch3_a(x))))
            ),
        )
        mixed = lucid.cat([b1, b2, b3], dim=1)
        out = cast(Tensor, self.proj(mixed))
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Block-B: Residual Inception-B (10× repeated, 17×17 × 1152)
# ---------------------------------------------------------------------------


class _BlockB(nn.Module):
    """Residual Inception-B: 17×17 × 1152 → 17×17 × 1152."""

    def __init__(self, in_channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        # branch1: 1×1(192)
        self.branch1 = _ConvBnReLU(in_channels, 192, 1)
        # branch2: 1×1(128) → 1×7(160) → 7×1(192)
        self.branch2_a = _ConvBnReLU(in_channels, 128, 1)
        self.branch2_b = nn.Conv2d(128, 160, (1, 7), padding=(0, 3), bias=False)
        self.branch2_b_bn = nn.BatchNorm2d(160)
        self.branch2_c = nn.Conv2d(160, 192, (7, 1), padding=(3, 0), bias=False)
        self.branch2_c_bn = nn.BatchNorm2d(192)
        # 1×1 projection: 192+192=384 → 1152
        self.proj = _ConvBn(384, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))

        t = cast(Tensor, self.branch2_a(x))
        t = F.relu(cast(Tensor, self.branch2_b_bn(cast(Tensor, self.branch2_b(t)))))
        b2 = F.relu(cast(Tensor, self.branch2_c_bn(cast(Tensor, self.branch2_c(t)))))

        mixed = lucid.cat([b1, b2], dim=1)
        out = cast(Tensor, self.proj(mixed))
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Reduction-B (17×17 → 8×8, 1152 → 2016)
# ---------------------------------------------------------------------------


class _ReductionB(nn.Module):
    """Reduction-B: 17×17 × 1152 → 8×8 × 2016.

    branch1: MaxPool s2            → 1152
    branch2: 1×1(256) → 3×3 s2   → 384
    branch3: 1×1(256) → 3×3 s2   → 288
    branch4: 1×1(256) → 3×3(288) → 3×3 s2 → 320
    total: 1152 + 384 + 288 + 320 = 2016
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.branch1 = nn.MaxPool2d(3, stride=2)
        self.branch2_a = _ConvBnReLU(in_channels, 256, 1)
        self.branch2_b = _ConvBnReLU(256, 384, 3, stride=2)
        self.branch3_a = _ConvBnReLU(in_channels, 256, 1)
        self.branch3_b = _ConvBnReLU(256, 288, 3, stride=2)
        self.branch4_a = _ConvBnReLU(in_channels, 256, 1)
        self.branch4_b = _ConvBnReLU(256, 288, 3, padding=1)
        self.branch4_c = _ConvBnReLU(288, 320, 3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
        b3 = cast(Tensor, self.branch3_b(cast(Tensor, self.branch3_a(x))))
        b4 = cast(
            Tensor,
            self.branch4_c(
                cast(Tensor, self.branch4_b(cast(Tensor, self.branch4_a(x))))
            ),
        )
        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Block-C: Residual Inception-C (5× repeated, 8×8 × 2016)
# ---------------------------------------------------------------------------


class _BlockC(nn.Module):
    """Residual Inception-C: 8×8 × 2016 → 8×8 × 2016.

    Actual channel count after Reduction-B with 1024-channel input:
    1024 (MaxPool) + 384 + 288 + 320 = 2016.
    """

    def __init__(self, in_channels: int, scale: float) -> None:
        super().__init__()
        self.scale = scale
        # branch1: 1×1(192)
        self.branch1 = _ConvBnReLU(in_channels, 192, 1)
        # branch2: 1×1(192) → [1×3(224), 3×1(256)] concat
        self.branch2_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch2_b1 = nn.Conv2d(192, 224, (1, 3), padding=(0, 1), bias=False)
        self.branch2_b1_bn = nn.BatchNorm2d(224)
        self.branch2_b2 = nn.Conv2d(192, 256, (3, 1), padding=(1, 0), bias=False)
        self.branch2_b2_bn = nn.BatchNorm2d(256)
        # 1×1 projection: 192+224+256=672 → in_channels
        self.proj = _ConvBn(672, in_channels, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))

        t2 = cast(Tensor, self.branch2_a(x))
        b2a = F.relu(
            cast(Tensor, self.branch2_b1_bn(cast(Tensor, self.branch2_b1(t2))))
        )
        b2b = F.relu(
            cast(Tensor, self.branch2_b2_bn(cast(Tensor, self.branch2_b2(t2))))
        )
        b2 = lucid.cat([b2a, b2b], dim=1)

        mixed = lucid.cat([b1, b2], dim=1)
        out = cast(Tensor, self.proj(mixed))
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Inception-ResNet v2 output dataclass
# ---------------------------------------------------------------------------


@dataclass
class InceptionResNetOutput:
    """Inception-ResNet v2 classification output."""

    logits: Tensor
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# InceptionResNetV2 backbone (task="base")
# ---------------------------------------------------------------------------


class InceptionResNetV2(PretrainedModel, BackboneMixin):
    """Inception-ResNet v2 feature extractor — outputs (B, 2016, 1, 1) for 299×299."""

    config_class: ClassVar[type[InceptionResNetConfig]] = InceptionResNetConfig
    base_model_prefix: ClassVar[str] = "inception_resnet_v2"

    def __init__(self, config: InceptionResNetConfig) -> None:
        super().__init__(config)
        s = config.scale
        self.stem = _build_stem(config.in_channels)

        # Block-A × 5
        self.block_a0 = _BlockA(384, s)
        self.block_a1 = _BlockA(384, s)
        self.block_a2 = _BlockA(384, s)
        self.block_a3 = _BlockA(384, s)
        self.block_a4 = _BlockA(384, s)

        # Reduction-A
        self.reduction_a = _ReductionA(384)

        # Block-B × 10
        self.block_b0 = _BlockB(1024, s)
        self.block_b1 = _BlockB(1024, s)
        self.block_b2 = _BlockB(1024, s)
        self.block_b3 = _BlockB(1024, s)
        self.block_b4 = _BlockB(1024, s)
        self.block_b5 = _BlockB(1024, s)
        self.block_b6 = _BlockB(1024, s)
        self.block_b7 = _BlockB(1024, s)
        self.block_b8 = _BlockB(1024, s)
        self.block_b9 = _BlockB(1024, s)

        # Reduction-B
        self.reduction_b = _ReductionB(1024)

        # Block-C × 5
        self.block_c0 = _BlockC(2016, s)
        self.block_c1 = _BlockC(2016, s)
        self.block_c2 = _BlockC(2016, s)
        self.block_c3 = _BlockC(2016, s)
        self.block_c4 = _BlockC(2016, s)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=384, reduction=8),
            FeatureInfo(stage=2, num_channels=1024, reduction=16),
            FeatureInfo(stage=3, num_channels=2016, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.block_a0(x))
        x = cast(Tensor, self.block_a1(x))
        x = cast(Tensor, self.block_a2(x))
        x = cast(Tensor, self.block_a3(x))
        x = cast(Tensor, self.block_a4(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.block_b0(x))
        x = cast(Tensor, self.block_b1(x))
        x = cast(Tensor, self.block_b2(x))
        x = cast(Tensor, self.block_b3(x))
        x = cast(Tensor, self.block_b4(x))
        x = cast(Tensor, self.block_b5(x))
        x = cast(Tensor, self.block_b6(x))
        x = cast(Tensor, self.block_b7(x))
        x = cast(Tensor, self.block_b8(x))
        x = cast(Tensor, self.block_b9(x))
        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.block_c0(x))
        x = cast(Tensor, self.block_c1(x))
        x = cast(Tensor, self.block_c2(x))
        x = cast(Tensor, self.block_c3(x))
        x = cast(Tensor, self.block_c4(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# InceptionResNetV2 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class InceptionResNetV2ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Inception-ResNet v2 with classification head."""

    config_class: ClassVar[type[InceptionResNetConfig]] = InceptionResNetConfig
    base_model_prefix: ClassVar[str] = "inception_resnet_v2"

    def __init__(self, config: InceptionResNetConfig) -> None:
        super().__init__(config)
        s = config.scale
        self.stem = _build_stem(config.in_channels)

        # Block-A × 5
        self.block_a0 = _BlockA(384, s)
        self.block_a1 = _BlockA(384, s)
        self.block_a2 = _BlockA(384, s)
        self.block_a3 = _BlockA(384, s)
        self.block_a4 = _BlockA(384, s)

        # Reduction-A
        self.reduction_a = _ReductionA(384)

        # Block-B × 10
        self.block_b0 = _BlockB(1024, s)
        self.block_b1 = _BlockB(1024, s)
        self.block_b2 = _BlockB(1024, s)
        self.block_b3 = _BlockB(1024, s)
        self.block_b4 = _BlockB(1024, s)
        self.block_b5 = _BlockB(1024, s)
        self.block_b6 = _BlockB(1024, s)
        self.block_b7 = _BlockB(1024, s)
        self.block_b8 = _BlockB(1024, s)
        self.block_b9 = _BlockB(1024, s)

        # Reduction-B
        self.reduction_b = _ReductionB(1024)

        # Block-C × 5
        self.block_c0 = _BlockC(2016, s)
        self.block_c1 = _BlockC(2016, s)
        self.block_c2 = _BlockC(2016, s)
        self.block_c3 = _BlockC(2016, s)
        self.block_c4 = _BlockC(2016, s)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(2016, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> InceptionResNetOutput:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.block_a0(x))
        x = cast(Tensor, self.block_a1(x))
        x = cast(Tensor, self.block_a2(x))
        x = cast(Tensor, self.block_a3(x))
        x = cast(Tensor, self.block_a4(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.block_b0(x))
        x = cast(Tensor, self.block_b1(x))
        x = cast(Tensor, self.block_b2(x))
        x = cast(Tensor, self.block_b3(x))
        x = cast(Tensor, self.block_b4(x))
        x = cast(Tensor, self.block_b5(x))
        x = cast(Tensor, self.block_b6(x))
        x = cast(Tensor, self.block_b7(x))
        x = cast(Tensor, self.block_b8(x))
        x = cast(Tensor, self.block_b9(x))
        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.block_c0(x))
        x = cast(Tensor, self.block_c1(x))
        x = cast(Tensor, self.block_c2(x))
        x = cast(Tensor, self.block_c3(x))
        x = cast(Tensor, self.block_c4(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return InceptionResNetOutput(logits=logits, loss=loss)
