"""ResNeSt backbone and classifier (Zhang et al., 2020).

Paper: "ResNeSt: Split-Attention Networks"

Key idea: Replace the 3×3 convolution in a ResNet Bottleneck with a
Split-Attention convolution that divides features into ``radix`` branches,
applies separate convolutions per branch, then recombines using an
attention-weighted sum (softmax over branches).
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.resnest._config import ResNeStConfig

# ---------------------------------------------------------------------------
# Split-Attention Convolution
# ---------------------------------------------------------------------------


class _SplitAttentionConv(nn.Module):
    """Split-Attention convolution: divide into ``radix`` branches and attend.

    Parameters
    ----------
    in_ch:              Input channels.
    out_ch:             Output channels per branch (total output = out_ch).
    stride:             Spatial stride for the depthwise convolution.
    radix:              Number of split branches (r in the paper).
    groups:             Cardinality groups (k).
    reduction_factor:   Channel reduction for the attention FC layers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        radix: int = 2,
        groups: int = 1,
        reduction_factor: int = 4,
    ) -> None:
        super().__init__()
        self._radix = radix
        self._out_ch = out_ch
        self._groups = groups

        inter_ch = out_ch * radix
        # Single conv that computes all radix branches at once
        self.conv = nn.Conv2d(
            in_ch,
            inter_ch,
            3,
            stride=stride,
            padding=1,
            groups=groups * radix,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(inter_ch)
        self.relu = nn.ReLU(inplace=True)

        # Attention FC layers (implemented as 1×1 convs for broadcast simplicity)
        att_ch = max(out_ch // reduction_factor, 32)
        self.fc1 = nn.Conv2d(out_ch, att_ch, 1, groups=groups, bias=False)
        self.bn_att = nn.BatchNorm2d(att_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(att_ch, out_ch * radix, 1, groups=groups)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # (B, out_ch*radix, H, W)
        out = cast(Tensor, self.relu(cast(Tensor, self.bn(cast(Tensor, self.conv(x))))))

        b, _, h, w = out.shape
        r = self._radix
        c = self._out_ch

        # (B, radix, out_ch, H, W)
        splits = out.reshape(b, r, c, h, w)
        # Sum over radix → (B, out_ch, H, W)
        gap = splits.sum(dim=1)
        # Global average pool → (B, out_ch, 1, 1)
        gap = gap.mean(dim=(2, 3), keepdim=True)

        # Attention: FC1 → FC2 → (B, radix*out_ch, 1, 1)
        attn = cast(
            Tensor,
            self.relu2(cast(Tensor, self.bn_att(cast(Tensor, self.fc1(gap))))),
        )
        attn = cast(Tensor, self.fc2(attn))
        # (B, radix, out_ch, 1, 1)
        attn = attn.reshape(b, r, c, 1, 1)

        if r > 1:
            attn = F.softmax(attn, dim=1)

        # Weighted sum: (B, radix, out_ch, H, W) * (B, radix, out_ch, 1, 1)
        result = (splits * attn).sum(dim=1)
        return result.reshape(b, c, h, w)


# ---------------------------------------------------------------------------
# ResNeSt Bottleneck
# ---------------------------------------------------------------------------


class _ResNeStBottleneck(nn.Module):
    """ResNet Bottleneck with Split-Attention 3×3 conv."""

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        radix: int = 2,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.downsample = downsample

        # 1×1 expand
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # Split-Attention 3×3
        self.conv2 = _SplitAttentionConv(
            out_channels, out_channels, stride=stride, radix=radix
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 1×1 project
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

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
    in_ch: int,
    width: int,
    num_blocks: int,
    stride: int,
    radix: int,
) -> tuple[nn.Sequential, int]:
    expansion = _ResNeStBottleneck.expansion
    final_ch = width * expansion

    downsample: nn.Module | None = None
    if stride != 1 or in_ch != final_ch:
        downsample = nn.Sequential(
            nn.Conv2d(in_ch, final_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(final_ch),
        )

    blocks: list[nn.Module] = [
        _ResNeStBottleneck(
            in_ch, width, stride=stride, radix=radix, downsample=downsample
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(_ResNeStBottleneck(final_ch, width, stride=1, radix=radix))

    return nn.Sequential(*blocks), final_ch


def _build_body(
    config: ResNeStConfig,
) -> tuple[
    nn.Sequential,
    nn.MaxPool2d,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    nn.Sequential,
    list[FeatureInfo],
]:
    stem_ch = 64
    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, stem_ch, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    widths = (64, 128, 256, 512)
    layers_cfg = config.layers
    radix = config.radix

    cur = stem_ch
    layer1, cur = _make_layer(cur, widths[0], layers_cfg[0], stride=1, radix=radix)
    layer2, cur = _make_layer(cur, widths[1], layers_cfg[1], stride=2, radix=radix)
    layer3, cur = _make_layer(cur, widths[2], layers_cfg[2], stride=2, radix=radix)
    layer4, cur = _make_layer(cur, widths[3], layers_cfg[3], stride=2, radix=radix)

    exp = _ResNeStBottleneck.expansion
    fi = [
        FeatureInfo(stage=1, num_channels=widths[0] * exp, reduction=4),
        FeatureInfo(stage=2, num_channels=widths[1] * exp, reduction=8),
        FeatureInfo(stage=3, num_channels=widths[2] * exp, reduction=16),
        FeatureInfo(stage=4, num_channels=widths[3] * exp, reduction=32),
    ]
    return stem, pool, layer1, layer2, layer3, layer4, fi


# ---------------------------------------------------------------------------
# ResNeSt backbone  (task="base")
# ---------------------------------------------------------------------------


class ResNeSt(PretrainedModel, BackboneMixin):
    """ResNeSt feature extractor — outputs (B, 2048, H/32, W/32)."""

    config_class: ClassVar[type[ResNeStConfig]] = ResNeStConfig
    base_model_prefix: ClassVar[str] = "resnest"

    def __init__(self, config: ResNeStConfig) -> None:
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
# ResNeSt for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ResNeStForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """ResNeSt with global average pooling + linear classification head."""

    config_class: ClassVar[type[ResNeStConfig]] = ResNeStConfig
    base_model_prefix: ClassVar[str] = "resnest"

    def __init__(self, config: ResNeStConfig) -> None:
        super().__init__(config)
        stem, pool, l1, l2, l3, l4, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        final_ch = 512 * _ResNeStBottleneck.expansion
        self._build_classifier(final_ch, config.num_classes, dropout=config.dropout)

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
