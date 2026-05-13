"""SENet backbone and classification head (Hu et al., 2017).

Paper: "Squeeze-and-Excitation Networks"
SE block: AdaptiveAvgPool2d(1) → Conv2d(C→rd_C,1×1) → ReLU
          → Conv2d(rd_C→C,1×1) → Sigmoid
The SE output is multiplied channel-wise with the block's feature map.
rd_channels = make_divisible(C / 16, divisor=8).
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import make_divisible as _make_divisible
from lucid.models.vision.senet._config import SENetConfig

# ---------------------------------------------------------------------------
# Squeeze-Excitation block (timm-accurate)
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    """Channel-wise squeeze-and-excitation gate — matches timm's SqueezeExcite.

    Uses Conv2d fc1/fc2 (kernel 1×1, bias=True) so the gate operates entirely
    in 4-D space without any flatten/reshape.
    rd_channels = make_divisible(channels / 16, divisor=8).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        rd_channels = _make_divisible(channels / reduction, divisor=8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Squeeze: (B, C, H, W) → (B, C, 1, 1)
        s = cast(Tensor, self.pool(x))
        # Excite: Conv2d operates on 4-D directly
        s = F.relu(cast(Tensor, self.fc1(s)))
        s = F.sigmoid(cast(Tensor, self.fc2(s)))
        # Broadcast-multiply (B, C, 1, 1) × (B, C, H, W)
        return x * s


# ---------------------------------------------------------------------------
# SE-BasicBlock
# ---------------------------------------------------------------------------


class _SEBasicBlock(nn.Module):
    """Two stacked 3×3 convolutions with SE gate — used in SE-ResNet-18/34."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = _SEBlock(out_channels, reduction=reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x

        out = cast(
            Tensor, self.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        )
        out = cast(Tensor, self.bn2(cast(Tensor, self.conv2(out))))
        out = cast(Tensor, self.se(out))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        out = out + identity
        return cast(Tensor, self.relu(out))


# ---------------------------------------------------------------------------
# SE-Bottleneck
# ---------------------------------------------------------------------------


class _SEBottleneck(nn.Module):
    """1×1 → 3×3 → 1×1 bottleneck with SE gate — used in SE-ResNet-50+."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.se = _SEBlock(out_channels * self.expansion, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x

        out = cast(
            Tensor, self.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        )
        out = cast(
            Tensor, self.relu(cast(Tensor, self.bn2(cast(Tensor, self.conv2(out)))))
        )
        out = cast(Tensor, self.bn3(cast(Tensor, self.conv3(out))))
        out = cast(Tensor, self.se(out))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        out = out + identity
        return cast(Tensor, self.relu(out))


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------

_BlockType = type[_SEBasicBlock] | type[_SEBottleneck]


def _make_layer(
    block_cls: _BlockType,
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
    reduction: int,
) -> tuple[nn.Sequential, int]:
    """Build one SE-ResNet stage. Returns (layer, new_in_channels)."""
    final_channels = out_channels * block_cls.expansion

    downsample: nn.Module | None = None
    if stride != 1 or in_channels != final_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    blocks: list[nn.Module] = [
        block_cls(
            in_channels,
            out_channels,
            stride=stride,
            downsample=downsample,
            reduction=reduction,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(block_cls(final_channels, out_channels, reduction=reduction))

    return nn.Sequential(*blocks), final_channels


def _build_body(
    config: SENetConfig,
) -> tuple[
    nn.Conv2d,
    nn.BatchNorm2d,
    nn.MaxPool2d,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    list[FeatureInfo],
]:
    """Build all sub-modules for an SE-ResNet body.

    Returns individual stem components (conv1, bn1) rather than a fused
    Sequential, so state-dict keys match timm's seresnet naming:
    ``conv1.weight``, ``bn1.*``, ``layer1.*``, …, ``layer4.*``.
    """
    block_cls: _BlockType = (
        _SEBasicBlock if config.block_type == "basic" else _SEBottleneck
    )
    stem_channels = 64
    hidden_sizes = (64, 128, 256, 512)

    conv1 = nn.Conv2d(
        config.in_channels, stem_channels, 7, stride=2, padding=3, bias=False
    )
    bn1 = nn.BatchNorm2d(stem_channels)
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    cur = stem_channels
    layer1, cur = _make_layer(
        block_cls, cur, hidden_sizes[0], config.layers[0], 1, config.reduction
    )
    layer2, cur = _make_layer(
        block_cls, cur, hidden_sizes[1], config.layers[1], 2, config.reduction
    )
    layer3, cur = _make_layer(
        block_cls, cur, hidden_sizes[2], config.layers[2], 2, config.reduction
    )
    layer4, cur = _make_layer(
        block_cls, cur, hidden_sizes[3], config.layers[3], 2, config.reduction
    )

    exp = block_cls.expansion
    feature_info = [
        FeatureInfo(stage=1, num_channels=hidden_sizes[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=hidden_sizes[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=hidden_sizes[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=hidden_sizes[3] * exp, reduction=32),
    ]
    return conv1, bn1, pool, layer1, layer2, layer3, layer4, feature_info


# ---------------------------------------------------------------------------
# SENet backbone (task="base")
# ---------------------------------------------------------------------------


class SENet(PretrainedModel, BackboneMixin):
    """SE-ResNet feature extractor — no classification head.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` of shape
    ``(B, C, H/32, W/32)`` from stage-4.

    State-dict keys follow timm's seresnet layout:
    ``conv1.*``, ``bn1.*``, ``layer1.*`` … ``layer4.*``.
    """

    config_class: ClassVar[type[SENetConfig]] = SENetConfig
    base_model_prefix: ClassVar[str] = "senet"

    def __init__(self, config: SENetConfig) -> None:
        super().__init__(config)
        conv1, bn1, pool, l1, l2, l3, l4, fi = _build_body(config)
        self.conv1 = conv1
        self.bn1 = bn1
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self._feature_info = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        x = cast(Tensor, self.maxpool(x))
        x = cast(Tensor, self.layer1(x))
        x = cast(Tensor, self.layer2(x))
        x = cast(Tensor, self.layer3(x))
        x = cast(Tensor, self.layer4(x))
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# SENet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class SENetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """SE-ResNet with global average pooling + linear classification head.

    State-dict keys follow timm's seresnet layout:
    ``conv1.*``, ``bn1.*``, ``layer1.*`` … ``layer4.*``, ``fc.*``.
    """

    config_class: ClassVar[type[SENetConfig]] = SENetConfig
    base_model_prefix: ClassVar[str] = "senet"

    def __init__(self, config: SENetConfig) -> None:
        super().__init__(config)
        conv1, bn1, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.conv1 = conv1
        self.bn1 = bn1
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4

        block_cls: _BlockType = (
            _SEBasicBlock if config.block_type == "basic" else _SEBottleneck
        )
        final_channels = 512 * block_cls.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(final_channels, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = F.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        x = cast(Tensor, self.maxpool(x))
        x = cast(Tensor, self.layer1(x))
        x = cast(Tensor, self.layer2(x))
        x = cast(Tensor, self.layer3(x))
        x = cast(Tensor, self.layer4(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.fc(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
