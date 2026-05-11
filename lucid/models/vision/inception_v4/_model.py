"""Inception v4 backbone and classifier (Szegedy et al., 2016).

Paper: "Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning"

Architecture overview (299×299 input):
    Stem: 7 convs → 384 channels (35×35 feature map)
    Inception-A × 4  → 384 channels (35×35)
    Reduction-A      → 1024 channels (17×17)
    Inception-B × 7  → 1024 channels (17×17)
    Reduction-B      → 1536 channels (8×8)
    Inception-C × 3  → 1536 channels (8×8)
    Head: AdaptiveAvgPool(1×1) → Dropout(0.2) → FC(1536, num_classes)
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
from lucid.models.vision.inception_v4._config import InceptionV4Config

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


# ---------------------------------------------------------------------------
# Stem (7 convs, 3 → 384)
# ---------------------------------------------------------------------------


def _build_v4_stem(in_channels: int) -> nn.Sequential:
    """Build Inception-v4 stem: input 299×299 → 35×35 × 384."""
    return nn.Sequential(
        _ConvBnReLU(in_channels, 32, 3, stride=2),  # 149×149
        _ConvBnReLU(32, 32, 3),  # 147×147
        _ConvBnReLU(32, 64, 3, padding=1),  # 147×147
        nn.MaxPool2d(3, stride=2),  # 73×73
        _ConvBnReLU(64, 80, 1),  # 73×73
        _ConvBnReLU(80, 192, 3),  # 71×71
        _ConvBnReLU(192, 384, 3, stride=2),  # 35×35
    )


# ---------------------------------------------------------------------------
# Inception-A (35×35, 4 branches → 384 output)
# ---------------------------------------------------------------------------


class _InceptionA(nn.Module):
    """Inception-A block: 4 branches → 384 channels (35×35)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1 (96)
        self.branch1 = _ConvBnReLU(in_channels, 96, 1)
        # branch2: 1×1(64) → 3×3(96)
        self.branch2_a = _ConvBnReLU(in_channels, 64, 1)
        self.branch2_b = _ConvBnReLU(64, 96, 3, padding=1)
        # branch3: 1×1(64) → 3×3(96) → 3×3(96)
        self.branch3_a = _ConvBnReLU(in_channels, 64, 1)
        self.branch3_b = _ConvBnReLU(64, 96, 3, padding=1)
        self.branch3_c = _ConvBnReLU(96, 96, 3, padding=1)
        # branch4: AvgPool → 1×1(96)
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, 96, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
        b3 = cast(
            Tensor,
            self.branch3_c(
                cast(Tensor, self.branch3_b(cast(Tensor, self.branch3_a(x))))
            ),
        )
        b4 = cast(Tensor, self.branch4_conv(cast(Tensor, self.branch4_pool(x))))
        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Reduction-A (35×35 → 17×17, 384 → 1024)
# ---------------------------------------------------------------------------


class _ReductionA(nn.Module):
    """Reduction-A: 35×35 × 384 → 17×17 × 1024."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 3×3 s2 (384)
        self.branch1 = _ConvBnReLU(in_channels, 384, 3, stride=2)
        # branch2: 1×1(192) → 3×3(224) → 3×3(256) s2
        self.branch2_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch2_b = _ConvBnReLU(192, 224, 3, padding=1)
        self.branch2_c = _ConvBnReLU(224, 256, 3, stride=2)
        # branch3: MaxPool s2 (keeps in_channels=384)
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
# Inception-B (17×17, 4 branches → 1024 output)
# ---------------------------------------------------------------------------


class _InceptionB(nn.Module):
    """Inception-B block: 4 branches → 1024 channels (17×17)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1 (384)
        self.branch1 = _ConvBnReLU(in_channels, 384, 1)
        # branch2: 1×1(192) → 1×7(224) → 7×1(256)
        self.branch2_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch2_b = nn.Conv2d(192, 224, (1, 7), padding=(0, 3), bias=False)
        self.branch2_b_bn = nn.BatchNorm2d(224)
        self.branch2_c = nn.Conv2d(224, 256, (7, 1), padding=(3, 0), bias=False)
        self.branch2_c_bn = nn.BatchNorm2d(256)
        # branch3: 1×1(192) → 7×1(192) → 1×7(224) → 7×1(224) → 1×7(256)
        self.branch3_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch3_b = nn.Conv2d(192, 192, (7, 1), padding=(3, 0), bias=False)
        self.branch3_b_bn = nn.BatchNorm2d(192)
        self.branch3_c = nn.Conv2d(192, 224, (1, 7), padding=(0, 3), bias=False)
        self.branch3_c_bn = nn.BatchNorm2d(224)
        self.branch3_d = nn.Conv2d(224, 224, (7, 1), padding=(3, 0), bias=False)
        self.branch3_d_bn = nn.BatchNorm2d(224)
        self.branch3_e = nn.Conv2d(224, 256, (1, 7), padding=(0, 3), bias=False)
        self.branch3_e_bn = nn.BatchNorm2d(256)
        # branch4: AvgPool → 1×1(128)
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, 128, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1(x))

        t = cast(Tensor, self.branch2_a(x))
        t = F.relu(cast(Tensor, self.branch2_b_bn(cast(Tensor, self.branch2_b(t)))))
        b2 = F.relu(cast(Tensor, self.branch2_c_bn(cast(Tensor, self.branch2_c(t)))))

        t = cast(Tensor, self.branch3_a(x))
        t = F.relu(cast(Tensor, self.branch3_b_bn(cast(Tensor, self.branch3_b(t)))))
        t = F.relu(cast(Tensor, self.branch3_c_bn(cast(Tensor, self.branch3_c(t)))))
        t = F.relu(cast(Tensor, self.branch3_d_bn(cast(Tensor, self.branch3_d(t)))))
        b3 = F.relu(cast(Tensor, self.branch3_e_bn(cast(Tensor, self.branch3_e(t)))))

        b4 = cast(Tensor, self.branch4_conv(cast(Tensor, self.branch4_pool(x))))

        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Reduction-B (17×17 → 8×8, 1024 → 1536)
# ---------------------------------------------------------------------------


class _ReductionB(nn.Module):
    """Reduction-B: 17×17 × 1024 → 8×8 × 1536."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1(192) → 3×3 s2 (192)
        self.branch1_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch1_b = _ConvBnReLU(192, 192, 3, stride=2)
        # branch2: 1×1(256) → 1×7(256) → 7×1(320) → 3×3 s2 (320)
        self.branch2_a = _ConvBnReLU(in_channels, 256, 1)
        self.branch2_b = nn.Conv2d(256, 256, (1, 7), padding=(0, 3), bias=False)
        self.branch2_b_bn = nn.BatchNorm2d(256)
        self.branch2_c = nn.Conv2d(256, 320, (7, 1), padding=(3, 0), bias=False)
        self.branch2_c_bn = nn.BatchNorm2d(320)
        self.branch2_d = _ConvBnReLU(320, 320, 3, stride=2)
        # branch3: MaxPool s2 (keeps 1024)
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b1 = cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))

        t = cast(Tensor, self.branch2_a(x))
        t = F.relu(cast(Tensor, self.branch2_b_bn(cast(Tensor, self.branch2_b(t)))))
        t = F.relu(cast(Tensor, self.branch2_c_bn(cast(Tensor, self.branch2_c(t)))))
        b2 = cast(Tensor, self.branch2_d(t))

        b3 = cast(Tensor, self.branch3(x))
        return lucid.cat([b1, b2, b3], dim=1)


# ---------------------------------------------------------------------------
# Inception-C (8×8, 4 branches → 1536 output)
# ---------------------------------------------------------------------------


class _InceptionC(nn.Module):
    """Inception-C block: 4 branches → 1536 channels (8×8)."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1 (256)
        self.branch1 = _ConvBnReLU(in_channels, 256, 1)
        # branch2: 1×1(384) → [1×3(256), 3×1(256)] concat
        self.branch2_a = _ConvBnReLU(in_channels, 384, 1)
        self.branch2_b1 = nn.Conv2d(384, 256, (1, 3), padding=(0, 1), bias=False)
        self.branch2_b1_bn = nn.BatchNorm2d(256)
        self.branch2_b2 = nn.Conv2d(384, 256, (3, 1), padding=(1, 0), bias=False)
        self.branch2_b2_bn = nn.BatchNorm2d(256)
        # branch3: 1×1(384) → 3×3(448) → 3×3(512) → [1×3(256), 3×1(256)] concat
        self.branch3_a = _ConvBnReLU(in_channels, 384, 1)
        self.branch3_b = _ConvBnReLU(384, 448, 3, padding=1)
        self.branch3_c = _ConvBnReLU(448, 512, 3, padding=1)
        self.branch3_d1 = nn.Conv2d(512, 256, (1, 3), padding=(0, 1), bias=False)
        self.branch3_d1_bn = nn.BatchNorm2d(256)
        self.branch3_d2 = nn.Conv2d(512, 256, (3, 1), padding=(1, 0), bias=False)
        self.branch3_d2_bn = nn.BatchNorm2d(256)
        # branch4: AvgPool → 1×1(256)
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, 256, 1)

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

        t3 = cast(
            Tensor,
            self.branch3_c(
                cast(Tensor, self.branch3_b(cast(Tensor, self.branch3_a(x))))
            ),
        )
        b3a = F.relu(
            cast(Tensor, self.branch3_d1_bn(cast(Tensor, self.branch3_d1(t3))))
        )
        b3b = F.relu(
            cast(Tensor, self.branch3_d2_bn(cast(Tensor, self.branch3_d2(t3))))
        )
        b3 = lucid.cat([b3a, b3b], dim=1)

        b4 = cast(Tensor, self.branch4_conv(cast(Tensor, self.branch4_pool(x))))

        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Inception v4 output dataclass
# ---------------------------------------------------------------------------


@dataclass
class InceptionV4Output:
    """Inception v4 classification output."""

    logits: Tensor
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# InceptionV4 backbone (task="base")
# ---------------------------------------------------------------------------


class InceptionV4(PretrainedModel, BackboneMixin):
    """Inception v4 feature extractor — outputs (B, 1536, 1, 1) for 299×299 inputs."""

    config_class: ClassVar[type[InceptionV4Config]] = InceptionV4Config
    base_model_prefix: ClassVar[str] = "inception_v4"

    def __init__(self, config: InceptionV4Config) -> None:
        super().__init__(config)
        self.stem = _build_v4_stem(config.in_channels)

        # Inception-A × 4
        self.inception_a0 = _InceptionA(384)
        self.inception_a1 = _InceptionA(384)
        self.inception_a2 = _InceptionA(384)
        self.inception_a3 = _InceptionA(384)

        # Reduction-A
        self.reduction_a = _ReductionA(384)

        # Inception-B × 7
        self.inception_b0 = _InceptionB(1024)
        self.inception_b1 = _InceptionB(1024)
        self.inception_b2 = _InceptionB(1024)
        self.inception_b3 = _InceptionB(1024)
        self.inception_b4 = _InceptionB(1024)
        self.inception_b5 = _InceptionB(1024)
        self.inception_b6 = _InceptionB(1024)

        # Reduction-B
        self.reduction_b = _ReductionB(1024)

        # Inception-C × 3
        self.inception_c0 = _InceptionC(1536)
        self.inception_c1 = _InceptionC(1536)
        self.inception_c2 = _InceptionC(1536)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=384, reduction=8),
            FeatureInfo(stage=2, num_channels=1024, reduction=16),
            FeatureInfo(stage=3, num_channels=1536, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception_a0(x))
        x = cast(Tensor, self.inception_a1(x))
        x = cast(Tensor, self.inception_a2(x))
        x = cast(Tensor, self.inception_a3(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.inception_b0(x))
        x = cast(Tensor, self.inception_b1(x))
        x = cast(Tensor, self.inception_b2(x))
        x = cast(Tensor, self.inception_b3(x))
        x = cast(Tensor, self.inception_b4(x))
        x = cast(Tensor, self.inception_b5(x))
        x = cast(Tensor, self.inception_b6(x))
        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.inception_c0(x))
        x = cast(Tensor, self.inception_c1(x))
        x = cast(Tensor, self.inception_c2(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# InceptionV4 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class InceptionV4ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Inception v4 with classification head."""

    config_class: ClassVar[type[InceptionV4Config]] = InceptionV4Config
    base_model_prefix: ClassVar[str] = "inception_v4"

    def __init__(self, config: InceptionV4Config) -> None:
        super().__init__(config)
        self.stem = _build_v4_stem(config.in_channels)

        # Inception-A × 4
        self.inception_a0 = _InceptionA(384)
        self.inception_a1 = _InceptionA(384)
        self.inception_a2 = _InceptionA(384)
        self.inception_a3 = _InceptionA(384)

        # Reduction-A
        self.reduction_a = _ReductionA(384)

        # Inception-B × 7
        self.inception_b0 = _InceptionB(1024)
        self.inception_b1 = _InceptionB(1024)
        self.inception_b2 = _InceptionB(1024)
        self.inception_b3 = _InceptionB(1024)
        self.inception_b4 = _InceptionB(1024)
        self.inception_b5 = _InceptionB(1024)
        self.inception_b6 = _InceptionB(1024)

        # Reduction-B
        self.reduction_b = _ReductionB(1024)

        # Inception-C × 3
        self.inception_c0 = _InceptionC(1536)
        self.inception_c1 = _InceptionC(1536)
        self.inception_c2 = _InceptionC(1536)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(1536, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> InceptionV4Output:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception_a0(x))
        x = cast(Tensor, self.inception_a1(x))
        x = cast(Tensor, self.inception_a2(x))
        x = cast(Tensor, self.inception_a3(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.inception_b0(x))
        x = cast(Tensor, self.inception_b1(x))
        x = cast(Tensor, self.inception_b2(x))
        x = cast(Tensor, self.inception_b3(x))
        x = cast(Tensor, self.inception_b4(x))
        x = cast(Tensor, self.inception_b5(x))
        x = cast(Tensor, self.inception_b6(x))
        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.inception_c0(x))
        x = cast(Tensor, self.inception_c1(x))
        x = cast(Tensor, self.inception_c2(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return InceptionV4Output(logits=logits, loss=loss)
