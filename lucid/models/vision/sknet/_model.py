"""SKNet backbone and classification head (Li et al., 2019).

Paper: "Selective Kernel Networks"
The SK block fuses two parallel conv branches (3×3 and dilated 3×3) via
channel-wise softmax attention derived from a global-average-pooled summary.

For SK-ResNeXt, both branches use grouped convolutions (cardinality > 1).
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.sknet._config import SKNetConfig

# ---------------------------------------------------------------------------
# SK block (replaces the 3×3 conv inside a bottleneck)
# ---------------------------------------------------------------------------

_BASE_WIDTH: int = 64


class _SKBlock(nn.Module):
    """Selective Kernel block: 2-branch multi-scale conv + attention fusion."""

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        groups: int = 1,
    ) -> None:
        super().__init__()
        # Branch 1: standard 3×3
        self.conv3 = nn.Conv2d(
            channels, channels, 3, padding=1, groups=groups, bias=False
        )
        self.bn3 = nn.BatchNorm2d(channels)
        # Branch 2: dilated 3×3 (effective 5×5 receptive field)
        self.conv5 = nn.Conv2d(
            channels, channels, 3, padding=2, dilation=2, groups=groups, bias=False
        )
        self.bn5 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        # Gating: squeeze → compact FC → softmax over 2 paths
        reduced = max(channels // reduction, 32)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, reduced, bias=False)
        self.bn_fc = nn.BatchNorm1d(reduced)
        # 2-path attention: (reduced → 2*channels) then reshaped
        self.attn = nn.Linear(reduced, channels * 2, bias=False)

        self._channels = channels

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        b = x.shape[0]
        c = self._channels

        # Compute two branch outputs
        u3 = cast(
            Tensor, self.relu(cast(Tensor, self.bn3(cast(Tensor, self.conv3(x)))))
        )
        u5 = cast(
            Tensor, self.relu(cast(Tensor, self.bn5(cast(Tensor, self.conv5(x)))))
        )

        # Fuse: element-wise sum then squeeze
        u = u3 + u5
        s = cast(Tensor, self.gap(u))  # (B, C, 1, 1)
        s = s.reshape(b, c)  # (B, C)

        # Compact representation
        z = cast(Tensor, self.relu(cast(Tensor, self.bn_fc(cast(Tensor, self.fc(s))))))

        # Attention weights: (B, 2*C) → (B, 2, C)
        attn_raw = cast(Tensor, self.attn(z))  # (B, 2*C)
        attn_raw = attn_raw.reshape(b, 2, c)  # (B, 2, C)
        attn = F.softmax(attn_raw, dim=1)  # softmax over 2 paths

        # Split attention weights for each branch: (B, C)
        a3 = attn[:, 0, :]  # (B, C)
        a5 = attn[:, 1, :]  # (B, C)

        # Reshape to (B, C, 1, 1) for broadcasting
        a3 = a3.reshape(b, c, 1, 1)
        a5 = a5.reshape(b, c, 1, 1)

        return u3 * a3 + u5 * a5


# ---------------------------------------------------------------------------
# SK Bottleneck block
# ---------------------------------------------------------------------------


class _SKBottleneck(nn.Module):
    """1×1 → SK(3×3/5×5) → 1×1 bottleneck.

    Uses standard ResNet-50 expansion=4.  The single 3×3 conv in each
    bottleneck is replaced by an SK block (two branches with different
    receptive fields).  ``cardinality`` (G in the paper) groups the SK
    branch convolutions; G=32 for SK-ResNet-50, G=32 for SK-ResNeXt-50.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        reduction: int = 16,
        cardinality: int = 32,
    ) -> None:
        super().__init__()
        width = out_channels

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        # SK replaces the 3×3 conv; stride applied via AvgPool after SK.
        self.sk = _SKBlock(width, reduction=reduction, groups=cardinality)
        self.stride_pool: nn.Module | None = (
            nn.AvgPool2d(stride, stride=stride) if stride > 1 else None
        )
        self.conv3 = nn.Conv2d(width, out_channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x

        out = cast(
            Tensor, self.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x)))))
        )
        out = cast(Tensor, self.sk(out))
        if self.stride_pool is not None:
            out = cast(Tensor, self.stride_pool(out))
        out = cast(Tensor, self.bn3(cast(Tensor, self.conv3(out))))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        out = out + identity
        return cast(Tensor, self.relu(out))


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------


def _make_layer(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
    reduction: int,
    cardinality: int,
) -> tuple[nn.Sequential, int]:
    """Build one SK-ResNet stage. Returns (layer, new_in_channels)."""
    final_channels = out_channels * _SKBottleneck.expansion

    downsample: nn.Module | None = None
    if stride != 1 or in_channels != final_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    blocks: list[nn.Module] = [
        _SKBottleneck(
            in_channels,
            out_channels,
            stride=stride,
            downsample=downsample,
            reduction=reduction,
            cardinality=cardinality,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            _SKBottleneck(
                final_channels,
                out_channels,
                reduction=reduction,
                cardinality=cardinality,
            )
        )

    return nn.Sequential(*blocks), final_channels


def _build_body(
    config: SKNetConfig,
) -> tuple[
    nn.Sequential,
    nn.MaxPool2d,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    list[FeatureInfo],
]:
    stem_channels = 64
    hidden_sizes = (64, 128, 256, 512)

    stem = nn.Sequential(
        nn.Conv2d(
            config.in_channels, stem_channels, 7, stride=2, padding=3, bias=False
        ),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    cur = stem_channels
    layer1, cur = _make_layer(
        cur, hidden_sizes[0], config.layers[0], 1, config.reduction, config.cardinality,
    )
    layer2, cur = _make_layer(
        cur, hidden_sizes[1], config.layers[1], 2, config.reduction, config.cardinality,
    )
    layer3, cur = _make_layer(
        cur, hidden_sizes[2], config.layers[2], 2, config.reduction, config.cardinality,
    )
    layer4, cur = _make_layer(
        cur, hidden_sizes[3], config.layers[3], 2, config.reduction, config.cardinality,
    )

    exp = _SKBottleneck.expansion
    feature_info = [
        FeatureInfo(stage=1, num_channels=hidden_sizes[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=hidden_sizes[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=hidden_sizes[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=hidden_sizes[3] * exp, reduction=32),
    ]
    return stem, pool, layer1, layer2, layer3, layer4, feature_info


# ---------------------------------------------------------------------------
# SKNet backbone (task="base")
# ---------------------------------------------------------------------------


class SKNet(PretrainedModel, BackboneMixin):
    """SK-ResNet feature extractor — no classification head.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` of shape
    ``(B, 2048, 7, 7)`` for 224×224 inputs.
    """

    config_class: ClassVar[type[SKNetConfig]] = SKNetConfig
    base_model_prefix: ClassVar[str] = "sknet"

    def __init__(self, config: SKNetConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, fi = _build_body(config)
        self.stem = stem
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
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.maxpool(x))
        x = cast(Tensor, self.layer1(x))
        x = cast(Tensor, self.layer2(x))
        x = cast(Tensor, self.layer3(x))
        x = cast(Tensor, self.layer4(x))
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# SKNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class SKNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """SK-ResNet with global average pooling + linear classification head."""

    config_class: ClassVar[type[SKNetConfig]] = SKNetConfig
    base_model_prefix: ClassVar[str] = "sknet"

    def __init__(self, config: SKNetConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4

        final_channels = 512 * _SKBottleneck.expansion  # 2048 (expansion=4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(final_channels, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.maxpool(x))
        x = cast(Tensor, self.layer1(x))
        x = cast(Tensor, self.layer2(x))
        x = cast(Tensor, self.layer3(x))
        x = cast(Tensor, self.layer4(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
