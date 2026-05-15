"""Inception-ResNet v2 backbone and classifier (Szegedy et al., 2016).

Paper: "Inception-v4, Inception-ResNet and the Impact of Residual Connections
        on Learning"

Architecture overview (299×299 input):
    Stem:        299×299×3  → 35×35×192  (5 named ConvNormAct)
    Mixed_5b:    35×35×192  → 35×35×320  (Inception-A pre-block)
    Block35 ×10  (residual Inception-A): 35×35×320,   scale=0.17
    Mixed_6a:    35×35×320  → 17×17×1088 (Reduction-A)
    Block17 ×20  (residual Inception-B): 17×17×1088,  scale=0.10
    Mixed_7a:    17×17×1088 →  8×8×2080  (Reduction-B)
    Block8  ×9   (residual Inception-C):  8×8×2080,   scale=0.20
    Block8  ×1   (no ReLU):               8×8×2080
    conv2d_7b:    8×8×2080  →  8×8×1536  (final projection)
    Head: AdaptiveAvgPool(1×1) → Dropout → classif(Linear)

State-dict naming matches timm inception_resnet_v2 exactly:
  conv2d_1a.conv/bn.*        ← stem Conv1
  conv2d_2a.conv/bn.*        ← stem Conv2
  conv2d_2b.conv/bn.*        ← stem Conv3
  conv2d_3b.conv/bn.*        ← stem Conv4
  conv2d_4a.conv/bn.*        ← stem Conv5
  mixed_5b.branch0.conv/bn.*
  mixed_5b.branch1.N.conv/bn.*      N = 0,1
  mixed_5b.branch2.N.conv/bn.*      N = 0,1,2
  mixed_5b.branch3.N.conv/bn.*      N = 1 (0 is AvgPool, no params)
  repeat.N.branch0.conv/bn.*
  repeat.N.branch1.M.conv/bn.*      M = 0,1
  repeat.N.branch2.M.conv/bn.*      M = 0,1,2
  repeat.N.conv2d.weight/bias       ← projection (has bias)
  mixed_6a.*
  repeat_1.N.*
  mixed_7a.branch0.M.*     M = 0,1
  mixed_7a.branch1.M.*     M = 0,1
  mixed_7a.branch2.M.*     M = 0,1,2
  repeat_2.N.*
  block8.*                  ← single final block (no relu)
  conv2d_7b.conv/bn.*
  classif.weight/bias       ← Linear head
"""

from dataclasses import dataclass
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, FeatureInfo
from lucid.models._output import BaseModelOutput
from lucid.models.vision.inception_resnet._config import InceptionResNetConfig

# ---------------------------------------------------------------------------
# ConvBnReLU helper — .conv and .bn sub-modules (matches timm ConvNormAct)
# ---------------------------------------------------------------------------


class _ConvBnReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU.

    Named sub-modules ``.conv`` and ``.bn`` to match timm's ConvNormAct keys.
    """

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
# Mixed_5b: 35×35×192 → 35×35×320  (Inception-A pre-block, no residual)
# ---------------------------------------------------------------------------


class _Mixed5b(nn.Module):
    """Initial Inception-style block: 192 → 96+64+96+64 = 320 channels.

    timm branch naming:
      branch0  : single ConvBnReLU
      branch1  : Sequential([0],[1])
      branch2  : Sequential([0],[1],[2])
      branch3  : Sequential([0]=AvgPool, [1]=ConvBnReLU)
    """

    def __init__(self) -> None:
        super().__init__()
        # branch0: 1×1 → 96
        self.branch0 = _ConvBnReLU(192, 96, 1)
        # branch1: 1×1(48) → 5×5(64)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(192, 48, 1),
            _ConvBnReLU(48, 64, 5, padding=2),
        )
        # branch2: 1×1(64) → 3×3(96) → 3×3(96)
        self.branch2 = nn.Sequential(
            _ConvBnReLU(192, 64, 1),
            _ConvBnReLU(64, 96, 3, padding=1),
            _ConvBnReLU(96, 96, 3, padding=1),
        )
        # branch3: AvgPool(3×3 s1 p1) → 1×1(64)
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            _ConvBnReLU(192, 64, 1),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2(x))
        b3 = cast(Tensor, self.branch3(x))
        return lucid.cat([b0, b1, b2, b3], dim=1)  # 96+64+96+64 = 320


# ---------------------------------------------------------------------------
# Block35: Residual Inception-A (×10, 35×35×320, scale=0.17)
# ---------------------------------------------------------------------------


class _Block35(nn.Module):
    """Residual Inception-A: 35×35×320 → 35×35×320.

    timm naming: branch0 (single), branch1 (Sequential[0,1]),
                 branch2 (Sequential[0,1,2]), conv2d (projection with bias).
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.branch0 = _ConvBnReLU(320, 32, 1)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(320, 32, 1),
            _ConvBnReLU(32, 32, 3, padding=1),
        )
        self.branch2 = nn.Sequential(
            _ConvBnReLU(320, 32, 1),
            _ConvBnReLU(32, 48, 3, padding=1),
            _ConvBnReLU(48, 64, 3, padding=1),
        )
        # timm uses `conv2d` (not `proj`) with bias=True
        self.conv2d = nn.Conv2d(128, 320, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2(x))
        mixed = lucid.cat([b0, b1, b2], dim=1)
        out = cast(Tensor, self.conv2d(mixed))
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Mixed_6a (Reduction-A): 35×35×320 → 17×17×1088
# ---------------------------------------------------------------------------


class _Mixed6a(nn.Module):
    """Reduction-A: 35×35×320 → 17×17×1088.

    branch0: 3×3 s2              → 384
    branch1: Sequential[0,1,2]   → 384
    branch2: MaxPool 3×3 s2      → 320 (pass-through, no params)
    total: 384 + 384 + 320 = 1088
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = _ConvBnReLU(320, 384, 3, stride=2)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(320, 256, 1),
            _ConvBnReLU(256, 256, 3, padding=1),
            _ConvBnReLU(256, 384, 3, stride=2),
        )
        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2(x))
        return lucid.cat([b0, b1, b2], dim=1)  # 384+384+320 = 1088


# ---------------------------------------------------------------------------
# Block17: Residual Inception-B (×20, 17×17×1088, scale=0.10)
# ---------------------------------------------------------------------------


class _Block17(nn.Module):
    """Residual Inception-B: 17×17×1088 → 17×17×1088.

    timm: branch0 (single), branch1 (Sequential[0,1,2]), conv2d.
    """

    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale
        self.branch0 = _ConvBnReLU(1088, 192, 1)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(1088, 128, 1),
            _ConvBnReLU(128, 160, (1, 7), padding=(0, 3)),
            _ConvBnReLU(160, 192, (7, 1), padding=(3, 0)),
        )
        self.conv2d = nn.Conv2d(384, 1088, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1(x))
        mixed = lucid.cat([b0, b1], dim=1)
        out = cast(Tensor, self.conv2d(mixed))
        return F.relu(x + self.scale * out)


# ---------------------------------------------------------------------------
# Mixed_7a (Reduction-B): 17×17×1088 → 8×8×2080
# ---------------------------------------------------------------------------


class _Mixed7a(nn.Module):
    """Reduction-B: 17×17×1088 → 8×8×2080.

    branch0: Sequential[0,1]     → 384
    branch1: Sequential[0,1]     → 288
    branch2: Sequential[0,1,2]   → 320
    branch3: MaxPool 3×3 s2      → 1088 (pass-through)
    total: 384 + 288 + 320 + 1088 = 2080
    """

    def __init__(self) -> None:
        super().__init__()
        self.branch0 = nn.Sequential(
            _ConvBnReLU(1088, 256, 1),
            _ConvBnReLU(256, 384, 3, stride=2),
        )
        self.branch1 = nn.Sequential(
            _ConvBnReLU(1088, 256, 1),
            _ConvBnReLU(256, 288, 3, stride=2),
        )
        self.branch2 = nn.Sequential(
            _ConvBnReLU(1088, 256, 1),
            _ConvBnReLU(256, 288, 3, padding=1),
            _ConvBnReLU(288, 320, 3, stride=2),
        )
        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1(x))
        b2 = cast(Tensor, self.branch2(x))
        b3 = cast(Tensor, self.branch3(x))
        return lucid.cat([b0, b1, b2, b3], dim=1)  # 384+288+320+1088 = 2080


# ---------------------------------------------------------------------------
# Block8: Residual Inception-C (×9 + ×1 no_relu, 8×8×2080, scale=0.20)
# ---------------------------------------------------------------------------


class _Block8(nn.Module):
    """Residual Inception-C: 8×8×2080 → 8×8×2080.

    timm: branch0 (single), branch1 (Sequential[0,1,2]), conv2d.
    The `no_relu` variant (final block) omits the activation on the output sum.
    """

    def __init__(self, scale: float, no_relu: bool = False) -> None:
        super().__init__()
        self.scale = scale
        self.no_relu = no_relu
        self.branch0 = _ConvBnReLU(2080, 192, 1)
        self.branch1 = nn.Sequential(
            _ConvBnReLU(2080, 192, 1),
            _ConvBnReLU(192, 224, (1, 3), padding=(0, 1)),
            _ConvBnReLU(224, 256, (3, 1), padding=(1, 0)),
        )
        self.conv2d = nn.Conv2d(448, 2080, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b0 = cast(Tensor, self.branch0(x))
        b1 = cast(Tensor, self.branch1(x))
        mixed = lucid.cat([b0, b1], dim=1)
        out = cast(Tensor, self.conv2d(mixed))
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
# Shared body builder (all stages as named attributes)
# ---------------------------------------------------------------------------


def _build_body(config: InceptionResNetConfig) -> dict[str, nn.Module]:
    """Return the ordered modules that make up the Inception-ResNet v2 body."""
    s_a = config.scale_a
    s_b = config.scale_b
    s_c = config.scale_c

    return {
        # Stem — 5 flat named ConvBnReLU (no MaxPool in state dict)
        "conv2d_1a": _ConvBnReLU(config.in_channels, 32, 3, stride=2),
        "conv2d_2a": _ConvBnReLU(32, 32, 3),
        "conv2d_2b": _ConvBnReLU(32, 64, 3, padding=1),
        # MaxPool 3×3 s2 — no params, stored separately in model
        "conv2d_3b": _ConvBnReLU(64, 80, 1),
        "conv2d_4a": _ConvBnReLU(80, 192, 3),
        # MaxPool 3×3 s2 — no params, stored separately in model
        # Inception body
        "mixed_5b": _Mixed5b(),
        "repeat": nn.Sequential(*[_Block35(scale=s_a) for _ in range(10)]),
        "mixed_6a": _Mixed6a(),
        "repeat_1": nn.Sequential(*[_Block17(scale=s_b) for _ in range(20)]),
        "mixed_7a": _Mixed7a(),
        "repeat_2": nn.Sequential(*[_Block8(scale=s_c) for _ in range(9)]),
        "block8": _Block8(scale=1.0, no_relu=True),
        "conv2d_7b": _ConvBnReLU(2080, 1536, 1),
    }


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
        body = _build_body(config)
        # Stem conv layers
        self.conv2d_1a: nn.Module = body["conv2d_1a"]
        self.conv2d_2a: nn.Module = body["conv2d_2a"]
        self.conv2d_2b: nn.Module = body["conv2d_2b"]
        self._pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b: nn.Module = body["conv2d_3b"]
        self.conv2d_4a: nn.Module = body["conv2d_4a"]
        self._pool2 = nn.MaxPool2d(3, stride=2)
        # Inception body
        self.mixed_5b: nn.Module = body["mixed_5b"]
        self.repeat: nn.Module = body["repeat"]
        self.mixed_6a: nn.Module = body["mixed_6a"]
        self.repeat_1: nn.Module = body["repeat_1"]
        self.mixed_7a: nn.Module = body["mixed_7a"]
        self.repeat_2: nn.Module = body["repeat_2"]
        self.block8: nn.Module = body["block8"]
        self.conv2d_7b: nn.Module = body["conv2d_7b"]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=320, reduction=8),
            FeatureInfo(stage=2, num_channels=1088, reduction=16),
            FeatureInfo(stage=3, num_channels=1536, reduction=32),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def _forward_stem(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.conv2d_1a(x))
        x = cast(Tensor, self.conv2d_2a(x))
        x = cast(Tensor, self.conv2d_2b(x))
        x = cast(Tensor, self._pool1(x))
        x = cast(Tensor, self.conv2d_3b(x))
        x = cast(Tensor, self.conv2d_4a(x))
        x = cast(Tensor, self._pool2(x))
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        x = self._forward_stem(x)
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


class InceptionResNetV2ForImageClassification(PretrainedModel):
    """Inception-ResNet v2 with classification head.

    Input: (B, 3, 299, 299)
    Output: InceptionResNetOutput with logits (B, num_classes)
    Head attribute name ``classif`` matches timm's inception_resnet_v2.
    """

    config_class: ClassVar[type[InceptionResNetConfig]] = InceptionResNetConfig
    base_model_prefix: ClassVar[str] = "inception_resnet_v2"

    def __init__(self, config: InceptionResNetConfig) -> None:
        super().__init__(config)
        body = _build_body(config)
        # Stem
        self.conv2d_1a: nn.Module = body["conv2d_1a"]
        self.conv2d_2a: nn.Module = body["conv2d_2a"]
        self.conv2d_2b: nn.Module = body["conv2d_2b"]
        self._pool1 = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b: nn.Module = body["conv2d_3b"]
        self.conv2d_4a: nn.Module = body["conv2d_4a"]
        self._pool2 = nn.MaxPool2d(3, stride=2)
        # Inception body
        self.mixed_5b: nn.Module = body["mixed_5b"]
        self.repeat: nn.Module = body["repeat"]
        self.mixed_6a: nn.Module = body["mixed_6a"]
        self.repeat_1: nn.Module = body["repeat_1"]
        self.mixed_7a: nn.Module = body["mixed_7a"]
        self.repeat_2: nn.Module = body["repeat_2"]
        self.block8: nn.Module = body["block8"]
        self.conv2d_7b: nn.Module = body["conv2d_7b"]

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if config.dropout > 0.0:
            self.dropout: nn.Module = nn.Dropout(config.dropout)
        else:
            self.dropout = nn.Identity()
        # Named `classif` to match timm
        self.classif = nn.Linear(1536, config.num_classes)

    def _forward_stem(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.conv2d_1a(x))
        x = cast(Tensor, self.conv2d_2a(x))
        x = cast(Tensor, self.conv2d_2b(x))
        x = cast(Tensor, self._pool1(x))
        x = cast(Tensor, self.conv2d_3b(x))
        x = cast(Tensor, self.conv2d_4a(x))
        x = cast(Tensor, self._pool2(x))
        return x

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> InceptionResNetOutput:
        x = self._forward_stem(x)
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
        x = cast(Tensor, self.dropout(x))
        logits = cast(Tensor, self.classif(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return InceptionResNetOutput(logits=logits, loss=loss)
