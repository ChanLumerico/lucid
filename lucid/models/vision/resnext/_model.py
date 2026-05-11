"""ResNeXt backbone and classification head (Xie et al., 2017).

Paper: "Aggregated Residual Transformations for Deep Neural Networks"
ResNeXt replaces the 3×3 conv in the ResNet bottleneck with a grouped
convolution.  The intermediate width is determined by:
    width = (width_per_group * out_channels // base_width) * cardinality
where base_width = 64 (the channel count at the first stage).
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.resnext._config import ResNeXtConfig

# ---------------------------------------------------------------------------
# ResNeXt bottleneck block
# ---------------------------------------------------------------------------

_BASE_WIDTH: int = 64


class _ResNeXtBottleneck(nn.Module):
    """1×1 → grouped 3×3 → 1×1 bottleneck with cardinality."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        cardinality: int = 32,
        width_per_group: int = 4,
    ) -> None:
        super().__init__()
        # Compute grouped-conv width
        width = (width_per_group * out_channels // _BASE_WIDTH) * cardinality

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
        out = cast(
            Tensor, self.relu(cast(Tensor, self.bn2(cast(Tensor, self.conv2(out)))))
        )
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
    cardinality: int,
    width_per_group: int,
) -> tuple[nn.Sequential, int]:
    """Build one ResNeXt stage. Returns (layer, new_in_channels)."""
    final_channels = out_channels * _ResNeXtBottleneck.expansion

    downsample: nn.Module | None = None
    if stride != 1 or in_channels != final_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    blocks: list[nn.Module] = [
        _ResNeXtBottleneck(
            in_channels,
            out_channels,
            stride=stride,
            downsample=downsample,
            cardinality=cardinality,
            width_per_group=width_per_group,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            _ResNeXtBottleneck(
                final_channels,
                out_channels,
                cardinality=cardinality,
                width_per_group=width_per_group,
            )
        )

    return nn.Sequential(*blocks), final_channels


def _build_body(
    config: ResNeXtConfig,
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
        cur,
        hidden_sizes[0],
        config.layers[0],
        1,
        config.cardinality,
        config.width_per_group,
    )
    layer2, cur = _make_layer(
        cur,
        hidden_sizes[1],
        config.layers[1],
        2,
        config.cardinality,
        config.width_per_group,
    )
    layer3, cur = _make_layer(
        cur,
        hidden_sizes[2],
        config.layers[2],
        2,
        config.cardinality,
        config.width_per_group,
    )
    layer4, cur = _make_layer(
        cur,
        hidden_sizes[3],
        config.layers[3],
        2,
        config.cardinality,
        config.width_per_group,
    )

    exp = _ResNeXtBottleneck.expansion
    feature_info = [
        FeatureInfo(stage=1, num_channels=hidden_sizes[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=hidden_sizes[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=hidden_sizes[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=hidden_sizes[3] * exp, reduction=32),
    ]
    return stem, pool, layer1, layer2, layer3, layer4, feature_info


# ---------------------------------------------------------------------------
# ResNeXt backbone (task="base")
# ---------------------------------------------------------------------------


class ResNeXt(PretrainedModel, BackboneMixin):
    """ResNeXt feature extractor — no classification head.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` of shape
    ``(B, 2048, 7, 7)`` for 224×224 inputs (same topology as ResNet-50+).
    """

    config_class: ClassVar[type[ResNeXtConfig]] = ResNeXtConfig
    base_model_prefix: ClassVar[str] = "resnext"

    def __init__(self, config: ResNeXtConfig) -> None:
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
# ResNeXt for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class ResNeXtForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """ResNeXt with global average pooling + linear classification head."""

    config_class: ClassVar[type[ResNeXtConfig]] = ResNeXtConfig
    base_model_prefix: ClassVar[str] = "resnext"

    def __init__(self, config: ResNeXtConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4

        final_channels = 512 * _ResNeXtBottleneck.expansion  # 2048
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
