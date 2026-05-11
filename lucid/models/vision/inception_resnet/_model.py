"""Inception-ResNet v2 backbone and classifier (Szegedy et al., 2016).

Paper: "Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning"

Architecture overview (299×299 input):
    Stem:        299×299×3  → 35×35×192  (5 conv + 2 maxpool)
    Mixed_5b:    35×35×192  → 35×35×320  (Inception-A pre-block)
    Block35 ×10  (residual Inception-A): 35×35×320,   scale=0.17
    Mixed_6a:    35×35×320  → 17×17×1088 (Reduction-A)
    Block17 ×20  (residual Inception-B): 17×17×1088,  scale=0.10
    Mixed_7a:    17×17×1088 →  8×8×2080  (Reduction-B)
    Block8  ×9   (residual Inception-C):  8×8×2080,   scale=0.20
    Block8  ×1   (no ReLU):               8×8×2080
    Conv2d_7b:    8×8×2080  →  8×8×1536  (final projection)
    Head: AdaptiveAvgPool(1×1) → Dropout → FC(1536, num_classes)

Channel totals that produce ~55.8M parameters match the Keras / timm reference.
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
# Stem: 299×299×3 → 35×35×192
# (5 conv layers + 2 max-pool, no Mixed_5b included here)
# ---------------------------------------------------------------------------


def _build_stem(in_channels: int) -> nn.Sequential:
    """Inception-ResNet-v2 stem: 299×299 → 35×35 × 192.

    Exact sequence (each conv has no padding unless noted):
        Conv 3×3 s2 →  32
        Conv 3×3    →  32
        Conv 3×3 p1 →  64
        MaxPool 3×3 s2
        Conv 1×1    →  80
        Conv 3×3    → 192
        MaxPool 3×3 s2
    """
    return nn.Sequential(
        _ConvBnReLU(in_channels, 32, 3, stride=2),  # 149×149×32
        _ConvBnReLU(32, 32, 3),  # 147×147×32
        _ConvBnReLU(32, 64, 3, padding=1),  # 147×147×64
        nn.MaxPool2d(3, stride=2),  #  73×73×64
        _ConvBnReLU(64, 80, 1),  #  73×73×80
        _ConvBnReLU(80, 192, 3),  #  71×71×192
        nn.MaxPool2d(3, stride=2),  #  35×35×192
    )


# ---------------------------------------------------------------------------
# Mixed_5b: 35×35×192 → 35×35×320  (Inception-A pre-block, no residual)
# ---------------------------------------------------------------------------


class _Mixed5b(nn.Module):
    """Initial Inception-style block: 192 → 96+64+96+64 = 320 channels."""

    def __init__(self) -> None:
        super().__init__()
        # branch0: 1×1 → 96
        self.branch0 = _ConvBnReLU(192, 96, 1)
        # branch1: 1×1(48) → 5×5(64)
        self.branch1_a = _ConvBnReLU(192, 48, 1)
        self.branch1_b = _ConvBnReLU(48, 64, 5, padding=2)
        # branch2: 1×1(64) → 3×3(96) → 3×3(96)
        self.branch2_a = _ConvBnReLU(192, 64, 1)
        self.branch2_b = _ConvBnReLU(64, 96, 3, padding=1)
        self.branch2_c = _ConvBnReLU(96, 96, 3, padding=1)
        # branch3: AvgPool(3×3 s1 p1) → 1×1(64)
        self.branch3_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch3_conv = _ConvBnReLU(192, 64, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))
        b2 = cast(
            Tensor,
            self.branch2_c(
                cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
            ),
        )
        b3 = cast(Tensor, self.branch3_conv(cast(Tensor, self.branch3_pool(x))))
        return lucid.cat([b0, b1, b2, b3], dim=1)  # 96+64+96+64 = 320


# ---------------------------------------------------------------------------
# Block35: Residual Inception-A (×10, 35×35×320, scale=0.17)
# ---------------------------------------------------------------------------


class _Block35(nn.Module):
    """Residual Inception-A: 35×35×320 → 35×35×320."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        # branch0: 1×1(32)
        self.branch0 = _ConvBnReLU(320, 32, 1)
        # branch1: 1×1(32) → 3×3(32)
        self.branch1_a = _ConvBnReLU(320, 32, 1)
        self.branch1_b = _ConvBnReLU(32, 32, 3, padding=1)
        # branch2: 1×1(32) → 3×3(48) → 3×3(64)
        self.branch2_a = _ConvBnReLU(320, 32, 1)
        self.branch2_b = _ConvBnReLU(32, 48, 3, padding=1)
        self.branch2_c = _ConvBnReLU(48, 64, 3, padding=1)
        # projection: 32+32+64=128 → 320 (no BN on projection per reference)
        self.proj = nn.Conv2d(128, 320, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))
        b2 = cast(
            Tensor,
            self.branch2_c(
                cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
            ),
        )
        mixed = lucid.cat([b0, b1, b2], dim=1)
        out = cast(Tensor, self.proj(mixed))
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Mixed_6a (Reduction-A): 35×35×320 → 17×17×1088
# ---------------------------------------------------------------------------


class _Mixed6a(nn.Module):
    """Reduction-A: 35×35×320 → 17×17×1088.

    branch0: 3×3 s2              → 384
    branch1: 1×1(256) → 3×3(256) → 3×3 s2 → 384
    branch2: MaxPool 3×3 s2      → 320 (pass-through)
    total: 384 + 384 + 320 = 1088
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _ConvBnReLU(320, 384, 3, stride=2)
        self.branch1_a = _ConvBnReLU(320, 256, 1)
        self.branch1_b = _ConvBnReLU(256, 256, 3, padding=1)
        self.branch1_c = _ConvBnReLU(256, 384, 3, stride=2)
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(
            Tensor,
            self.branch1_c(
                cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))
            ),
        )
        b2 = cast(Tensor, self.branch2(x))
        return lucid.cat([b0, b1, b2], dim=1)  # 384+384+320 = 1088


# ---------------------------------------------------------------------------
# Block17: Residual Inception-B (×20, 17×17×1088, scale=0.10)
# ---------------------------------------------------------------------------


class _Block17(nn.Module):
    """Residual Inception-B: 17×17×1088 → 17×17×1088."""

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        # branch0: 1×1(192)
        self.branch0 = _ConvBnReLU(1088, 192, 1)
        # branch1: 1×1(128) → 1×7(160) → 7×1(192)
        self.branch1_a = _ConvBnReLU(1088, 128, 1)
        self.branch1_b = _ConvBnReLU(128, 160, (1, 7), padding=(0, 3))
        self.branch1_c = _ConvBnReLU(160, 192, (7, 1), padding=(3, 0))
        # projection: 192+192=384 → 1088
        self.proj = nn.Conv2d(384, 1088, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(
            Tensor,
            self.branch1_c(
                cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))
            ),
        )
        mixed = lucid.cat([b0, b1], dim=1)
        out = cast(Tensor, self.proj(mixed))
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Mixed_7a (Reduction-B): 17×17×1088 → 8×8×2080
# ---------------------------------------------------------------------------


class _Mixed7a(nn.Module):
    """Reduction-B: 17×17×1088 → 8×8×2080.

    branch0: 1×1(256) → 3×3 s2              → 384
    branch1: 1×1(256) → 3×3 s2              → 288
    branch2: 1×1(256) → 3×3(288) → 3×3 s2  → 320
    branch3: MaxPool 3×3 s2                  → 1088 (pass-through)
    total: 384 + 288 + 320 + 1088 = 2080
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0_a = _ConvBnReLU(1088, 256, 1)
        self.branch0_b = _ConvBnReLU(256, 384, 3, stride=2)
        self.branch1_a = _ConvBnReLU(1088, 256, 1)
        self.branch1_b = _ConvBnReLU(256, 288, 3, stride=2)
        self.branch2_a = _ConvBnReLU(1088, 256, 1)
        self.branch2_b = _ConvBnReLU(256, 288, 3, padding=1)
        self.branch2_c = _ConvBnReLU(288, 320, 3, stride=2)
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0_b(cast(Tensor, self.branch0_a(x))))
        b1 = cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))
        b2 = cast(
            Tensor,
            self.branch2_c(
                cast(Tensor, self.branch2_b(cast(Tensor, self.branch2_a(x))))
            ),
        )
        b3 = cast(Tensor, self.branch3(x))
        return lucid.cat([b0, b1, b2, b3], dim=1)  # 384+288+320+1088 = 2080


# ---------------------------------------------------------------------------
# Block8: Residual Inception-C (×9 + ×1 no_relu, 8×8×2080, scale=0.20)
# ---------------------------------------------------------------------------


class _Block8(nn.Module):
    """Residual Inception-C: 8×8×2080 → 8×8×2080.

    branch0: 1×1(192)
    branch1: 1×1(192) → 1×3(224) → 3×1(256)  [sequential, not parallel]
    mixed: 192+256 = 448 → proj 2080
    """

    def __init__(self, scale: float, no_relu: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        # branch0: 1×1(192)
        self.branch0 = _ConvBnReLU(2080, 192, 1)
        # branch1: 1×1(192) → 1×3(224) → 3×1(256)
        self.branch1_a = _ConvBnReLU(2080, 192, 1)
        self.branch1_b = _ConvBnReLU(192, 224, (1, 3), padding=(0, 1))
        self.branch1_c = _ConvBnReLU(224, 256, (3, 1), padding=(1, 0))
        # projection: 192+256=448 → 2080
        self.proj = nn.Conv2d(448, 2080, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(
            Tensor,
            self.branch1_c(
                cast(Tensor, self.branch1_b(cast(Tensor, self.branch1_a(x))))
            ),
        )
        mixed = lucid.cat([b0, b1], dim=1)
        out = cast(Tensor, self.proj(mixed))
        result = x + self.scale * out
        if self.no_relu:
            return result
        return F.relu(result)


# ---------------------------------------------------------------------------
# Inception-ResNet v2 output dataclass
# ---------------------------------------------------------------------------


@dataclass
class InceptionResNetOutput:
    """Inception-ResNet v2 classification output."""

    logits: Tensor
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# Shared forward body (builds and runs all stages)
# ---------------------------------------------------------------------------


def _build_stages(config: InceptionResNetConfig) -> nn.Sequential:
    """Not used directly — stages are stored as named attributes for BackboneMixin."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# InceptionResNetV2 backbone (task="base")
# ---------------------------------------------------------------------------


class InceptionResNetV2(PretrainedModel, BackboneMixin):
    """Inception-ResNet v2 feature extractor.

    Final feature map before classifier: (B, 1536, 1, 1) for 299×299 input.
    """

    config_class: ClassVar[type[InceptionResNetConfig]] = InceptionResNetConfig
    base_model_prefix: ClassVar[str] = "inception_resnet_v2"

    def __init__(self, config: InceptionResNetConfig) -> None:
        super().__init__(config)
        s_a = config.scale_a  # Block35  default 0.17
        s_b = config.scale_b  # Block17  default 0.10
        s_c = config.scale_c  # Block8   default 0.20

        # Stem: 299×299×in_channels → 35×35×192
        self.stem = _build_stem(config.in_channels)

        # Mixed_5b: 35×35×192 → 35×35×320
        self.mixed_5b = _Mixed5b()

        # Block35 × 10: 35×35×320
        self.repeat = nn.Sequential(*[_Block35(scale=s_a) for _ in range(10)])

        # Mixed_6a (Reduction-A): 35×35×320 → 17×17×1088
        self.mixed_6a = _Mixed6a()

        # Block17 × 20: 17×17×1088
        self.repeat_1 = nn.Sequential(*[_Block17(scale=s_b) for _ in range(20)])

        # Mixed_7a (Reduction-B): 17×17×1088 → 8×8×2080
        self.mixed_7a = _Mixed7a()

        # Block8 × 9 (with ReLU): 8×8×2080
        self.repeat_2 = nn.Sequential(*[_Block8(scale=s_c) for _ in range(9)])

        # Block8 × 1 (no ReLU): 8×8×2080
        self.block8 = _Block8(scale=1.0, no_relu=True)

        # Final projection: 2080 → 1536
        self.conv2d_7b = _ConvBnReLU(2080, 1536, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=320, reduction=8),
            FeatureInfo(stage=2, num_channels=1088, reduction=16),
            FeatureInfo(stage=3, num_channels=1536, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.mixed_5b(x))
        x = cast(Tensor, self.repeat(x))
        x = cast(Tensor, self.mixed_6a(x))
        x = cast(Tensor, self.repeat_1(x))
        x = cast(Tensor, self.mixed_7a(x))
        x = cast(Tensor, self.repeat_2(x))
        x = cast(Tensor, self.block8(x))
        x = cast(Tensor, self.conv2d_7b(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# InceptionResNetV2 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class InceptionResNetV2ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Inception-ResNet v2 with classification head.

    Input: (B, 3, 299, 299)
    Output: InceptionResNetOutput with logits (B, num_classes)
    """

    config_class: ClassVar[type[InceptionResNetConfig]] = InceptionResNetConfig
    base_model_prefix: ClassVar[str] = "inception_resnet_v2"

    def __init__(self, config: InceptionResNetConfig) -> None:
        super().__init__(config)
        s_a = config.scale_a
        s_b = config.scale_b
        s_c = config.scale_c

        # Stem: 299×299×in_channels → 35×35×192
        self.stem = _build_stem(config.in_channels)

        # Mixed_5b: 35×35×192 → 35×35×320
        self.mixed_5b = _Mixed5b()

        # Block35 × 10: 35×35×320
        self.repeat = nn.Sequential(*[_Block35(scale=s_a) for _ in range(10)])

        # Mixed_6a (Reduction-A): 35×35×320 → 17×17×1088
        self.mixed_6a = _Mixed6a()

        # Block17 × 20: 17×17×1088
        self.repeat_1 = nn.Sequential(*[_Block17(scale=s_b) for _ in range(20)])

        # Mixed_7a (Reduction-B): 17×17×1088 → 8×8×2080
        self.mixed_7a = _Mixed7a()

        # Block8 × 9 (with ReLU): 8×8×2080
        self.repeat_2 = nn.Sequential(*[_Block8(scale=s_c) for _ in range(9)])

        # Block8 × 1 (no ReLU): 8×8×2080
        self.block8 = _Block8(scale=1.0, no_relu=True)

        # Final projection: 2080 → 1536
        self.conv2d_7b = _ConvBnReLU(2080, 1536, 1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(1536, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> InceptionResNetOutput:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.mixed_5b(x))
        x = cast(Tensor, self.repeat(x))
        x = cast(Tensor, self.mixed_6a(x))
        x = cast(Tensor, self.repeat_1(x))
        x = cast(Tensor, self.mixed_7a(x))
        x = cast(Tensor, self.repeat_2(x))
        x = cast(Tensor, self.block8(x))
        x = cast(Tensor, self.conv2d_7b(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return InceptionResNetOutput(logits=logits, loss=loss)
