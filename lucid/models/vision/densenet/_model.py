"""DenseNet backbone and classifier (Huang et al., 2016).

Paper: "Densely Connected Convolutional Networks"

Key idea: each layer receives feature maps from ALL preceding layers in the block
(dense connection), then contributes its own k feature maps.  This encourages
feature reuse and keeps the model compact.

Architecture:
    Stem     : Conv7×7-s2(→num_init_features) → BN → ReLU → MaxPool3×3-s2
    Block 1  : DenseBlock(layers[0]) → Transition (½ channels, AvgPool-s2)
    Block 2  : DenseBlock(layers[1]) → Transition
    Block 3  : DenseBlock(layers[2]) → Transition
    Block 4  : DenseBlock(layers[3])
    Head     : BN → ReLU → AdaptiveAvgPool(1×1) → FC

Each DenseLayer (bottleneck variant):
    BN → ReLU → Conv1×1(bn_size×k) → BN → ReLU → Conv3×3(k) [→ Dropout]

Transition block:
    BN → ReLU → Conv1×1(½ in_channels) → AvgPool2×2-s2
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.densenet._config import DenseNetConfig

# ---------------------------------------------------------------------------
# Dense layer (bottleneck)
# ---------------------------------------------------------------------------


class _DenseLayer(nn.Module):
    """Single dense layer: BN-ReLU-Conv1×1-BN-ReLU-Conv3×3."""

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        inter_features = bn_size * growth_rate
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, inter_features, 1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_features)
        self.conv2 = nn.Conv2d(inter_features, growth_rate, 3, padding=1, bias=False)
        self.drop = nn.Dropout(p=dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = cast(Tensor, self.conv1(F.relu(cast(Tensor, self.norm1(x)))))
        out = cast(Tensor, self.conv2(F.relu(cast(Tensor, self.norm2(out)))))
        if self.drop is not None:
            out = cast(Tensor, self.drop(out))
        # Dense connection: concat input with new features along channel dim
        return lucid.cat([x, out], dim=1)


# ---------------------------------------------------------------------------
# Dense block
# ---------------------------------------------------------------------------


class _DenseBlock(nn.Module):
    """N stacked _DenseLayers; each layer receives all previous outputs."""

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers.append(
                _DenseLayer(
                    num_input_features + i * growth_rate,
                    growth_rate,
                    bn_size,
                    dropout_rate,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        for layer in self.layers:
            x = cast(Tensor, layer(x))
        return x


# ---------------------------------------------------------------------------
# Transition block
# ---------------------------------------------------------------------------


class _Transition(nn.Module):
    """BN-ReLU-Conv1×1-AvgPool — halves channels and spatial resolution."""

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, 1, bias=False)
        self.pool = nn.AvgPool2d(2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.conv(F.relu(cast(Tensor, self.norm(x)))))
        return cast(Tensor, self.pool(x))


# ---------------------------------------------------------------------------
# Shared builder: stem + blocks + transitions
# ---------------------------------------------------------------------------


def _build_densenet(
    cfg: DenseNetConfig,
) -> tuple[nn.Sequential, nn.ModuleList, nn.ModuleList, nn.BatchNorm2d, int]:
    """Returns (stem, blocks, transitions, final_norm, num_features).

    ``transitions`` has len(blocks) - 1 elements (no transition after last block).
    """
    stem = nn.Sequential(
        nn.Conv2d(
            cfg.in_channels, cfg.num_init_features, 7, stride=2, padding=3, bias=False
        ),
        nn.BatchNorm2d(cfg.num_init_features),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
    )

    blocks: list[nn.Module] = []
    transitions: list[nn.Module] = []
    num_features = cfg.num_init_features

    for i, num_layers in enumerate(cfg.block_config):
        block = _DenseBlock(
            num_layers, num_features, cfg.growth_rate, cfg.bn_size, cfg.dropout_rate
        )
        blocks.append(block)
        num_features += num_layers * cfg.growth_rate

        if i < len(cfg.block_config) - 1:
            out_features = num_features // 2
            transitions.append(_Transition(num_features, out_features))
            num_features = out_features

    final_norm = nn.BatchNorm2d(num_features)
    return (
        stem,
        nn.ModuleList(blocks),
        nn.ModuleList(transitions),
        final_norm,
        num_features,
    )


# ---------------------------------------------------------------------------
# DenseNet backbone  (task="base")
# ---------------------------------------------------------------------------


class DenseNet(PretrainedModel, BackboneMixin):
    """DenseNet feature extractor — outputs final dense-block activations.

    ``forward_features`` returns shape ``(B, C, 1, 1)`` for 224×224 inputs
    after the final BN-ReLU and AdaptiveAvgPool2d(1×1).
    """

    config_class: ClassVar[type[DenseNetConfig]] = DenseNetConfig
    base_model_prefix: ClassVar[str] = "densenet"

    def __init__(self, config: DenseNetConfig) -> None:
        super().__init__(config)
        stem, blocks, transitions, norm, num_features = _build_densenet(config)
        self.stem = stem
        self.blocks = blocks
        self.transitions = transitions
        self.final_norm = norm
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._num_features = num_features

        # feature_info: one entry per block (after each dense block)
        fi: list[FeatureInfo] = []
        nf = config.num_init_features
        reduction = 4
        for i, nl in enumerate(config.block_config):
            nf += nl * config.growth_rate
            ch = nf if i == len(config.block_config) - 1 else nf // 2
            if i < len(config.block_config) - 1:
                reduction *= 2
            fi.append(FeatureInfo(stage=i + 1, num_channels=ch, reduction=reduction))
        self._feature_info = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        for i, block in enumerate(self.blocks):
            x = cast(Tensor, block(x))
            if i < len(self.transitions):
                x = cast(Tensor, self.transitions[i](x))
        x = F.relu(cast(Tensor, self.final_norm(x)))
        return cast(Tensor, self.avgpool(x))

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# DenseNet for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class DenseNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """DenseNet with global average pool + linear classification head."""

    config_class: ClassVar[type[DenseNetConfig]] = DenseNetConfig
    base_model_prefix: ClassVar[str] = "densenet"

    def __init__(self, config: DenseNetConfig) -> None:
        super().__init__(config)
        stem, blocks, transitions, norm, num_features = _build_densenet(config)
        self.stem = stem
        self.blocks = blocks
        self.transitions = transitions
        self.final_norm = norm
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(num_features, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        for i, block in enumerate(self.blocks):
            x = cast(Tensor, block(x))
            if i < len(self.transitions):
                x = cast(Tensor, self.transitions[i](x))
        x = F.relu(cast(Tensor, self.final_norm(x)))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
