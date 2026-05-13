"""SE-ResNeXt backbone and classifier.

Combines ResNeXt-style grouped convolution with Squeeze-and-Excitation gates,
following the formulation from:

  "Squeeze-and-Excitation Networks" (Hu et al., 2018) applied to ResNeXt blocks.

SE-ResNeXt bottleneck:
  Conv(in_ch, width, 1)                          → BN → ReLU
  Conv(width, width, 3, groups=cardinality)       → BN → ReLU
  Conv(width, out_ch*4, 1)                        → BN
  SE: AvgPool → Conv(C, C//se_reduction) → ReLU → Conv → Sigmoid → scale
  + downsample shortcut (if stride != 1 or in_ch != out_ch*4)
  ReLU

where  width = int(planes * (base_width / 64.0)) * cardinality
       planes = the bottleneck "base" channel count (pre-expansion)
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.se_resnext._config import SEResNeXtConfig

_BASE_WIDTH: int = 64  # reference channel count for width formula

# ---------------------------------------------------------------------------
# SE block
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    """Channel-wise Squeeze-and-Excitation gate.

    Uses Conv2d fc1/fc2 (1×1, bias=True) so the gate operates entirely in
    4-D space without any flatten/reshape.
    """

    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        rd_channels: int = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        s = cast(Tensor, self.pool(x))
        s = F.relu(cast(Tensor, self.fc1(s)))
        s = F.sigmoid(cast(Tensor, self.fc2(s)))
        return x * s


# ---------------------------------------------------------------------------
# SE-ResNeXt bottleneck block
# ---------------------------------------------------------------------------


class _SEResNeXtBottleneck(nn.Module):
    """1×1 → grouped 3×3 → 1×1 bottleneck with SE gate."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        cardinality: int = 32,
        base_width: int = 4,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        # Compute grouped-conv width
        width: int = int(planes * (base_width / _BASE_WIDTH)) * cardinality
        out_channels: int = planes * self.expansion

        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(
            width,
            width,
            3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.se = _SEBlock(out_channels, se_reduction)
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


def _make_layer(
    in_channels: int,
    planes: int,
    num_blocks: int,
    stride: int,
    cardinality: int,
    base_width: int,
    se_reduction: int,
) -> tuple[nn.Sequential, int]:
    """Build one SE-ResNeXt stage.  Returns (layer, new_in_channels)."""
    out_channels: int = planes * _SEResNeXtBottleneck.expansion

    downsample: nn.Module | None = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    blocks: list[nn.Module] = [
        _SEResNeXtBottleneck(
            in_channels,
            planes,
            stride=stride,
            downsample=downsample,
            cardinality=cardinality,
            base_width=base_width,
            se_reduction=se_reduction,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            _SEResNeXtBottleneck(
                out_channels,
                planes,
                cardinality=cardinality,
                base_width=base_width,
                se_reduction=se_reduction,
            )
        )

    return nn.Sequential(*blocks), out_channels


def _build_body(
    config: SEResNeXtConfig,
) -> tuple[
    nn.Sequential,
    nn.MaxPool2d,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    list[FeatureInfo],
]:
    stem_channels: int = 64
    planes_per_stage: tuple[int, int, int, int] = (64, 128, 256, 512)

    stem = nn.Sequential(
        nn.Conv2d(
            config.in_channels,
            stem_channels,
            7,
            stride=2,
            padding=3,
            bias=False,
        ),
        nn.BatchNorm2d(stem_channels),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    cur: int = stem_channels
    layer1, cur = _make_layer(
        cur,
        planes_per_stage[0],
        config.layers[0],
        1,
        config.cardinality,
        config.base_width,
        config.se_reduction,
    )
    layer2, cur = _make_layer(
        cur,
        planes_per_stage[1],
        config.layers[1],
        2,
        config.cardinality,
        config.base_width,
        config.se_reduction,
    )
    layer3, cur = _make_layer(
        cur,
        planes_per_stage[2],
        config.layers[2],
        2,
        config.cardinality,
        config.base_width,
        config.se_reduction,
    )
    layer4, cur = _make_layer(
        cur,
        planes_per_stage[3],
        config.layers[3],
        2,
        config.cardinality,
        config.base_width,
        config.se_reduction,
    )

    exp: int = _SEResNeXtBottleneck.expansion
    feature_info: list[FeatureInfo] = [
        FeatureInfo(stage=1, num_channels=planes_per_stage[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=planes_per_stage[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=planes_per_stage[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=planes_per_stage[3] * exp, reduction=32),
    ]
    return stem, pool, layer1, layer2, layer3, layer4, feature_info


# ---------------------------------------------------------------------------
# SE-ResNeXt backbone (task="base")
# ---------------------------------------------------------------------------


class SEResNeXt(PretrainedModel, BackboneMixin):
    """SE-ResNeXt feature extractor — outputs (B, 2048, H/32, W/32)."""

    config_class: ClassVar[type[SEResNeXtConfig]] = SEResNeXtConfig
    base_model_prefix: ClassVar[str] = "se_resnext"

    def __init__(self, config: SEResNeXtConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, fi = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self._feature_info: list[FeatureInfo] = fi

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
# SE-ResNeXt for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class SEResNeXtForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """SE-ResNeXt with global average pooling + linear classification head."""

    config_class: ClassVar[type[SEResNeXtConfig]] = SEResNeXtConfig
    base_model_prefix: ClassVar[str] = "se_resnext"

    def __init__(self, config: SEResNeXtConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4

        final_channels: int = 512 * _SEResNeXtBottleneck.expansion  # 2048
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(
            final_channels, config.num_classes, dropout=config.dropout
        )

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
