"""Inception v3 backbone and classifier (Szegedy et al., 2015).

Paper: "Rethinking the Inception Architecture for Computer Vision"

Architecture overview (299×299 input):
    Stem   : Conv3×3-s2 → Conv3×3 → Conv3×3-p1 → MaxPool-s2
             → Conv1×1 → Conv3×3 → MaxPool-s2
    InceptionA × 3: 4-branch concat (pool_features=32, 32, 64)
    InceptionB (Reduction-A): 3-branch, stride-2 reduction
    InceptionC × 4: factorized n×1 / 1×n blocks (n=7)
    InceptionD (Reduction-B): 3-branch, stride-2 reduction
    InceptionE × 2: expanded branch with 1×3 / 3×1 splits
    Auxiliary classifier attaches after InceptionC[1]
    Head: AdaptiveAvgPool(1×1) → Dropout(0.5) → FC(2048, num_classes)
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
from lucid.models.vision.inception._config import InceptionConfig

# ---------------------------------------------------------------------------
# Shared Conv-BN-ReLU helper
# ---------------------------------------------------------------------------


class _ConvBnReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int = 0,
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
# Inception A — 3× repeated with varying pool_features
# ---------------------------------------------------------------------------


class _InceptionA(nn.Module):
    """Inception-A block (factorized 5×5 into two 3×3)."""

    def __init__(self, in_channels: int, pool_features: int) -> None:
        super().__init__()
        # branch1: 1×1
        self.branch1 = _ConvBnReLU(in_channels, 64, 1)
        # branch2: 1×1 → 5×5 (implemented as two 3×3)
        self.branch2_a = _ConvBnReLU(in_channels, 48, 1)
        self.branch2_b = _ConvBnReLU(48, 64, 5, padding=2)
        # branch3: 1×1 → 3×3 → 3×3
        self.branch3_a = _ConvBnReLU(in_channels, 64, 1)
        self.branch3_b = _ConvBnReLU(64, 96, 3, padding=1)
        self.branch3_c = _ConvBnReLU(96, 96, 3, padding=1)
        # branch4: AvgPool → 1×1
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, pool_features, 1)

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
# Inception B — Reduction-A (stride-2 reduction 35×35 → 17×17)
# ---------------------------------------------------------------------------


class _InceptionB(nn.Module):
    """Reduction-A block: reduces 35×35 → 17×17."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 3×3 stride=2
        self.branch1 = _ConvBnReLU(in_channels, 384, 3, stride=2)
        # branch2: 1×1 → 3×3 → 3×3 stride=2
        self.branch2_a = _ConvBnReLU(in_channels, 64, 1)
        self.branch2_b = _ConvBnReLU(64, 96, 3, padding=1)
        self.branch2_c = _ConvBnReLU(96, 96, 3, stride=2)
        # branch3: MaxPool stride=2 (passthrough)
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
# Inception C — 4× factorized n×1 and 1×n (n=7)
# ---------------------------------------------------------------------------


class _InceptionC(nn.Module):
    """Inception-C block with 1×n / n×1 factorization (n=7)."""

    def __init__(self, in_channels: int, channels_7x7: int) -> None:
        super().__init__()
        c7 = channels_7x7
        # branch1: 1×1
        self.branch1 = _ConvBnReLU(in_channels, 192, 1)
        # branch2: 1×1 → 1×7 → 7×1
        self.branch2_a = _ConvBnReLU(in_channels, c7, 1)
        self.branch2_b = nn.Conv2d(c7, c7, (1, 7), padding=(0, 3), bias=False)
        self.branch2_b_bn = nn.BatchNorm2d(c7)
        self.branch2_c = nn.Conv2d(c7, 192, (7, 1), padding=(3, 0), bias=False)
        self.branch2_c_bn = nn.BatchNorm2d(192)
        # branch3: 1×1 → 7×1 → 1×7 → 7×1 → 1×7
        self.branch3_a = _ConvBnReLU(in_channels, c7, 1)
        self.branch3_b = nn.Conv2d(c7, c7, (7, 1), padding=(3, 0), bias=False)
        self.branch3_b_bn = nn.BatchNorm2d(c7)
        self.branch3_c = nn.Conv2d(c7, c7, (1, 7), padding=(0, 3), bias=False)
        self.branch3_c_bn = nn.BatchNorm2d(c7)
        self.branch3_d = nn.Conv2d(c7, c7, (7, 1), padding=(3, 0), bias=False)
        self.branch3_d_bn = nn.BatchNorm2d(c7)
        self.branch3_e = nn.Conv2d(c7, 192, (1, 7), padding=(0, 3), bias=False)
        self.branch3_e_bn = nn.BatchNorm2d(192)
        # branch4: AvgPool → 1×1
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, 192, 1)

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
# Inception D — Reduction-B (stride-2 reduction 17×17 → 8×8)
# ---------------------------------------------------------------------------


class _InceptionD(nn.Module):
    """Reduction-B block: reduces 17×17 → 8×8."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1 → 3×3 stride=2
        self.branch1_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch1_b = _ConvBnReLU(192, 320, 3, stride=2)
        # branch2: 1×1 → 1×7 → 7×1 → 3×3 stride=2
        self.branch2_a = _ConvBnReLU(in_channels, 192, 1)
        self.branch2_b = nn.Conv2d(192, 192, (1, 7), padding=(0, 3), bias=False)
        self.branch2_b_bn = nn.BatchNorm2d(192)
        self.branch2_c = nn.Conv2d(192, 192, (7, 1), padding=(3, 0), bias=False)
        self.branch2_c_bn = nn.BatchNorm2d(192)
        self.branch2_d = _ConvBnReLU(192, 192, 3, stride=2)
        # branch3: MaxPool stride=2
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
# Inception E — 2× expanded branches with 1×3/3×1 splits
# ---------------------------------------------------------------------------


class _InceptionE(nn.Module):
    """Inception-E block: expanded branches with parallel 1×3 / 3×1 convs."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # branch1: 1×1
        self.branch1 = _ConvBnReLU(in_channels, 320, 1)
        # branch2: 1×1(384) → [1×3(384), 3×1(384)] concat
        self.branch2_a = _ConvBnReLU(in_channels, 384, 1)
        self.branch2_b1 = nn.Conv2d(384, 384, (1, 3), padding=(0, 1), bias=False)
        self.branch2_b1_bn = nn.BatchNorm2d(384)
        self.branch2_b2 = nn.Conv2d(384, 384, (3, 1), padding=(1, 0), bias=False)
        self.branch2_b2_bn = nn.BatchNorm2d(384)
        # branch3: 1×1(448) → 3×3(384) → [1×3(384), 3×1(384)] concat
        self.branch3_a = _ConvBnReLU(in_channels, 448, 1)
        self.branch3_b = _ConvBnReLU(448, 384, 3, padding=1)
        self.branch3_c1 = nn.Conv2d(384, 384, (1, 3), padding=(0, 1), bias=False)
        self.branch3_c1_bn = nn.BatchNorm2d(384)
        self.branch3_c2 = nn.Conv2d(384, 384, (3, 1), padding=(1, 0), bias=False)
        self.branch3_c2_bn = nn.BatchNorm2d(384)
        # branch4: AvgPool → 1×1(192)
        self.branch4_pool = nn.AvgPool2d(3, stride=1, padding=1)
        self.branch4_conv = _ConvBnReLU(in_channels, 192, 1)

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

        t3 = cast(Tensor, self.branch3_b(cast(Tensor, self.branch3_a(x))))
        b3a = F.relu(
            cast(Tensor, self.branch3_c1_bn(cast(Tensor, self.branch3_c1(t3))))
        )
        b3b = F.relu(
            cast(Tensor, self.branch3_c2_bn(cast(Tensor, self.branch3_c2(t3))))
        )
        b3 = lucid.cat([b3a, b3b], dim=1)

        b4 = cast(Tensor, self.branch4_conv(cast(Tensor, self.branch4_pool(x))))

        return lucid.cat([b1, b2, b3, b4], dim=1)


# ---------------------------------------------------------------------------
# Auxiliary classifier
# ---------------------------------------------------------------------------


class _InceptionAux(nn.Module):
    """Auxiliary classifier for Inception v3 (attaches after InceptionC[1])."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.avgpool = nn.AvgPool2d(5, stride=3)
        self.conv = _ConvBnReLU(in_channels, 128, 1)
        self.adapt_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.avgpool(x))
        x = cast(Tensor, self.conv(x))
        x = cast(Tensor, self.adapt_pool(x))
        x = x.flatten(1)
        return cast(Tensor, self.fc(x))


# ---------------------------------------------------------------------------
# Inception v3 output dataclass
# ---------------------------------------------------------------------------


@dataclass
class InceptionV3Output:
    """Inception v3 output — includes optional auxiliary logits for training."""

    logits: Tensor
    aux_logits: Tensor | None = None
    loss: Tensor | None = None


# ---------------------------------------------------------------------------
# Stem builder
# ---------------------------------------------------------------------------


def _build_inception_stem(in_channels: int) -> nn.Sequential:
    return nn.Sequential(
        _ConvBnReLU(in_channels, 32, 3, stride=2),
        _ConvBnReLU(32, 32, 3),
        _ConvBnReLU(32, 64, 3, padding=1),
        nn.MaxPool2d(3, stride=2),
        _ConvBnReLU(64, 80, 1),
        _ConvBnReLU(80, 192, 3),
        nn.MaxPool2d(3, stride=2),
    )


# ---------------------------------------------------------------------------
# InceptionV3 backbone (task="base")
# ---------------------------------------------------------------------------


class InceptionV3(PretrainedModel, BackboneMixin):
    """Inception v3 feature extractor — outputs (B, 2048, 1, 1) for 299×299 inputs."""

    config_class: ClassVar[type[InceptionConfig]] = InceptionConfig
    base_model_prefix: ClassVar[str] = "inception_v3"

    def __init__(self, config: InceptionConfig) -> None:
        super().__init__(config)
        self.stem = _build_inception_stem(config.in_channels)

        # InceptionA × 3 (pool_features = 32, 32, 64)
        # InceptionA always outputs 64+64+96+pool_features channels
        # a0: 192→256 (pool=32), a1: 256→256 (pool=32), a2: 256→288 (pool=64)
        self.inception_a0 = _InceptionA(192, pool_features=32)
        self.inception_a1 = _InceptionA(256, pool_features=32)
        self.inception_a2 = _InceptionA(256, pool_features=64)

        # Reduction-A (InceptionB)
        self.reduction_a = _InceptionB(288)

        # InceptionC × 4 (channels_7x7 = 128, 160, 160, 192)
        self.inception_c0 = _InceptionC(768, channels_7x7=128)
        self.inception_c1 = _InceptionC(768, channels_7x7=160)
        self.inception_c2 = _InceptionC(768, channels_7x7=160)
        self.inception_c3 = _InceptionC(768, channels_7x7=192)

        # Reduction-B (InceptionD)
        self.reduction_b = _InceptionD(768)

        # InceptionE × 2
        self.inception_e0 = _InceptionE(1280)
        self.inception_e1 = _InceptionE(2048)

        # Final pool
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=288, reduction=8),
            FeatureInfo(stage=2, num_channels=768, reduction=16),
            FeatureInfo(stage=3, num_channels=2048, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception_a0(x))
        x = cast(Tensor, self.inception_a1(x))
        x = cast(Tensor, self.inception_a2(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.inception_c0(x))
        x = cast(Tensor, self.inception_c1(x))
        x = cast(Tensor, self.inception_c2(x))
        x = cast(Tensor, self.inception_c3(x))
        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.inception_e0(x))
        x = cast(Tensor, self.inception_e1(x))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# InceptionV3 for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class InceptionV3ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Inception v3 with classification head and optional auxiliary classifier."""

    config_class: ClassVar[type[InceptionConfig]] = InceptionConfig
    base_model_prefix: ClassVar[str] = "inception_v3"

    def __init__(self, config: InceptionConfig) -> None:
        super().__init__(config)
        self.stem = _build_inception_stem(config.in_channels)

        # InceptionA × 3
        # a0: 192→256, a1: 256→256, a2: 256→288
        self.inception_a0 = _InceptionA(192, pool_features=32)
        self.inception_a1 = _InceptionA(256, pool_features=32)
        self.inception_a2 = _InceptionA(256, pool_features=64)

        # Reduction-A (input: 288 channels = 64+64+96+64)
        self.reduction_a = _InceptionB(288)

        # InceptionC × 4
        self.inception_c0 = _InceptionC(768, channels_7x7=128)
        self.inception_c1 = _InceptionC(768, channels_7x7=160)
        self.inception_c2 = _InceptionC(768, channels_7x7=160)
        self.inception_c3 = _InceptionC(768, channels_7x7=192)

        # Reduction-B
        self.reduction_b = _InceptionD(768)

        # InceptionE × 2
        self.inception_e0 = _InceptionE(1280)
        self.inception_e1 = _InceptionE(2048)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=config.dropout)
        self._build_classifier(2048, config.num_classes)

        # Auxiliary classifier (attaches after inception_c1)
        if config.aux_logits:
            self.aux: nn.Module = _InceptionAux(768, config.num_classes)
        else:
            self.aux = nn.Identity()

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> InceptionV3Output:
        cfg = self.config
        assert isinstance(cfg, InceptionConfig)
        use_aux = cfg.aux_logits and self.training

        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.inception_a0(x))
        x = cast(Tensor, self.inception_a1(x))
        x = cast(Tensor, self.inception_a2(x))
        x = cast(Tensor, self.reduction_a(x))
        x = cast(Tensor, self.inception_c0(x))
        x = cast(Tensor, self.inception_c1(x))

        aux_out: Tensor | None = None
        if use_aux and isinstance(self.aux, _InceptionAux):
            aux_out = cast(Tensor, self.aux(x))

        x = cast(Tensor, self.inception_c2(x))
        x = cast(Tensor, self.inception_c3(x))
        x = cast(Tensor, self.reduction_b(x))
        x = cast(Tensor, self.inception_e0(x))
        x = cast(Tensor, self.inception_e1(x))
        x = cast(Tensor, self.avgpool(x))
        x = cast(Tensor, self.drop(x.flatten(1)))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            if aux_out is not None:
                loss = loss + 0.4 * F.cross_entropy(aux_out, labels)

        return InceptionV3Output(logits=logits, aux_logits=aux_out, loss=loss)
