"""ResNet backbone and classification head."""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.resnet._config import ResNetConfig


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class _BasicBlock(nn.Module):
    """Two stacked 3×3 convolutions — used in ResNet-18/34."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> Tensor:
        identity = x

        out = cast(Tensor, self.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x))))))
        out = cast(Tensor, self.bn2(cast(Tensor, self.conv2(out))))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        out = out + identity
        return cast(Tensor, self.relu(out))


class _Bottleneck(nn.Module):
    """1×1 → 3×3 → 1×1 bottleneck — used in ResNet-50/101/152."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # out_channels is the bottleneck width; final width is out_channels * 4
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
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> Tensor:
        identity = x

        out = cast(Tensor, self.relu(cast(Tensor, self.bn1(cast(Tensor, self.conv1(x))))))
        out = cast(Tensor, self.relu(cast(Tensor, self.bn2(cast(Tensor, self.conv2(out))))))
        out = cast(Tensor, self.bn3(cast(Tensor, self.conv3(out))))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        out = out + identity
        return cast(Tensor, self.relu(out))


# ---------------------------------------------------------------------------
# Shared ResNet stem + body builder
# ---------------------------------------------------------------------------


def _make_layer(
    block_cls: type[_BasicBlock] | type[_Bottleneck],
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int = 1,
) -> tuple[nn.Sequential, int]:
    """Build one ResNet stage. Returns (layer, new_in_channels)."""
    expansion = block_cls.expansion
    final_channels = out_channels * expansion

    downsample: nn.Module | None = None
    if stride != 1 or in_channels != final_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(final_channels),
        )

    layers: list[nn.Module] = [block_cls(in_channels, out_channels, stride, downsample)]
    for _ in range(1, num_blocks):
        layers.append(block_cls(final_channels, out_channels))

    return nn.Sequential(*layers), final_channels


def _build_body(
    config: ResNetConfig,
) -> tuple[
    nn.Sequential,
    nn.MaxPool2d,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    list[FeatureInfo],
]:
    block_cls: type[_BasicBlock] | type[_Bottleneck] = (
        _BasicBlock if config.block_type == "basic" else _Bottleneck
    )
    sc = config.stem_channels
    hs = config.hidden_sizes

    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, sc, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(sc),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    cur = sc
    layer1, cur = _make_layer(block_cls, cur, hs[0], config.layers[0])
    layer2, cur = _make_layer(block_cls, cur, hs[1], config.layers[1], stride=2)
    layer3, cur = _make_layer(block_cls, cur, hs[2], config.layers[2], stride=2)
    layer4, cur = _make_layer(block_cls, cur, hs[3], config.layers[3], stride=2)

    exp = block_cls.expansion
    feature_info = [
        FeatureInfo(stage=1, num_channels=hs[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=hs[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=hs[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=hs[3] * exp, reduction=32),
    ]
    return stem, pool, layer1, layer2, layer3, layer4, feature_info


def _zero_init_residual(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, _Bottleneck):
            if m.bn3.weight is not None:
                nn.init.constant_(m.bn3.weight, 0.0)
        elif isinstance(m, _BasicBlock):
            if m.bn2.weight is not None:
                nn.init.constant_(m.bn2.weight, 0.0)


# ---------------------------------------------------------------------------
# ResNet backbone (task="base")
# ---------------------------------------------------------------------------


class ResNet(PretrainedModel, BackboneMixin):
    """ResNet feature extractor — no classification head.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` of shape
    ``(B, C, H/32, W/32)`` from stage-4.
    """

    config_class: ClassVar[type[ResNetConfig]] = ResNetConfig
    base_model_prefix: ClassVar[str] = "resnet"

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, fi = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self._feature_info = fi

        if config.zero_init_residual:
            _zero_init_residual(self)

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

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> BaseModelOutput:
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# ResNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class ResNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """ResNet with global average pooling + linear classification head."""

    config_class: ClassVar[type[ResNetConfig]] = ResNetConfig
    base_model_prefix: ClassVar[str] = "resnet"

    def __init__(self, config: ResNetConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4

        block_cls: type[_BasicBlock] | type[_Bottleneck] = (
            _BasicBlock if config.block_type == "basic" else _Bottleneck
        )
        final_channels = config.hidden_sizes[-1] * block_cls.expansion
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(final_channels, config.num_classes, dropout=config.dropout)

        if config.zero_init_residual:
            _zero_init_residual(self)

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
