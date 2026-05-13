"""ResNeSt backbone and classifier (Zhang et al., 2020).

Paper: "ResNeSt: Split-Attention Networks" — https://arxiv.org/abs/2004.08955

Architecture overview for ResNeSt-50 (radix=2, groups=1, avd=True, avg_down=True):

  Deep Stem : Conv(3→32, k=3, s=2) → BN → ReLU
              Conv(32→32, k=3, s=1) → BN → ReLU
              Conv(32→64, k=3, s=1) → BN → ReLU
  MaxPool   : 3×3, stride=2
  Stage 1–4 : ResNeStBottleneck blocks with Split-Attention conv
  Head      : AdaptiveAvgPool(1×1) → Flatten → Linear

Split-Attention convolution (SplitAttn):
  - A single grouped conv produces ``radix`` branches at once
  - Global average pool over the branch sum produces an attention vector
  - Branches are recombined via softmax attention weights
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import make_divisible as _make_divisible
from lucid.models.vision.resnest._config import ResNeStConfig

# ---------------------------------------------------------------------------
# RadixSoftmax
# ---------------------------------------------------------------------------


class _RadixSoftmax(nn.Module):
    """Per-radix softmax (or sigmoid when radix==1)."""

    def __init__(self, radix: int, groups: int) -> None:
        super().__init__()
        self._radix = radix
        self._groups = groups

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        batch = x.shape[0]
        if self._radix > 1:
            # (B, groups, radix, C) → permute → (B, radix, groups, C)
            x = x.reshape(batch, self._groups, self._radix, -1).permute(0, 2, 1, 3)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = F.sigmoid(x)
        return x


# ---------------------------------------------------------------------------
# Split-Attention convolution
# ---------------------------------------------------------------------------


class _SplitAttn(nn.Module):
    """Split-Attention (Splat) convolution.

    Parameters
    ----------
    in_channels:    Input channels.
    out_channels:   Output channels (= input channels in ResNeSt bottleneck).
    kernel_size:    Convolution kernel size.
    stride:         Spatial stride (applied to the main grouped conv).
    padding:        Spatial padding.
    dilation:       Dilation for the main conv.
    groups:         Cardinality groups.
    radix:          Number of split branches.
    rd_ratio:       Attention channel reduction ratio.
    rd_divisor:     Channel divisibility constraint.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        radix: int = 2,
        rd_ratio: float = 0.25,
        rd_divisor: int = 8,
    ) -> None:
        super().__init__()
        self._radix = radix
        mid_chs = out_channels * radix
        attn_chs = _make_divisible(
            in_channels * radix * rd_ratio,
            divisor=rd_divisor,
            min_value=32,
        )

        # Single grouped conv producing all radix branches simultaneously
        self.conv = nn.Conv2d(
            in_channels,
            mid_chs,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups * radix,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(mid_chs)
        self.act0 = nn.ReLU(inplace=True)

        # Attention path — 1×1 convs with default bias=True
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = nn.BatchNorm2d(attn_chs)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = _RadixSoftmax(radix, groups)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.conv(x))
        x = cast(Tensor, self.bn0(x))
        x = cast(Tensor, self.act0(x))

        b, rc, h, w = x.shape
        r = self._radix

        if r > 1:
            # (B, radix, C, H, W)
            x = x.reshape(b, r, rc // r, h, w)
            x_gap = x.sum(dim=1)  # (B, C, H, W)
        else:
            x_gap = x

        # Global average pool → (B, C, 1, 1)
        x_gap = x_gap.mean((2, 3), keepdim=True)
        x_gap = cast(Tensor, self.fc1(x_gap))
        x_gap = cast(Tensor, self.bn1(x_gap))
        x_gap = cast(Tensor, self.act1(x_gap))
        x_attn = cast(Tensor, self.fc2(x_gap))

        x_attn = cast(Tensor, self.rsoftmax(x_attn)).reshape(b, -1, 1, 1)

        if r > 1:
            # x: (B, radix, C, H, W) ; x_attn: (B, radix*C, 1, 1) → (B, radix, C, 1, 1)
            out = (x * x_attn.reshape(b, r, rc // r, 1, 1)).sum(dim=1)
        else:
            out = x * x_attn

        return out


# ---------------------------------------------------------------------------
# ResNeSt Bottleneck
# ---------------------------------------------------------------------------


class _ResNeStBottleneck(nn.Module):
    """ResNeSt Bottleneck block.

    Parameters
    ----------
    inplanes:   Input channels.
    planes:     Base width (output = planes * 4 due to expansion=4).
    stride:     Spatial stride for the block.
    downsample: Optional downsampling module for the skip connection.
    radix:      Number of split branches in SplitAttn.
    groups:     Cardinality.
    avd:        Use averaged downsampling (AvgPool before/after SplitAttn).
    avd_first:  Place the AvgPool before (True) or after (False) SplitAttn.
    is_first:   True for the first block of the first stage (triggers avd even
                at stride=1 to match the reference implementation).
    dilation:   Dilation for the SplitAttn conv.
    """

    expansion: ClassVar[int] = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        radix: int = 2,
        groups: int = 1,
        avd: bool = True,
        avd_first: bool = False,
        is_first: bool = False,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        group_width = int(planes * groups)

        # Determine avd pooling stride
        if avd and (stride > 1 or is_first):
            avd_stride = stride
            conv_stride = 1
        else:
            avd_stride = 0
            conv_stride = stride

        # 1×1 expand
        self.conv1 = nn.Conv2d(inplanes, group_width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.act1 = nn.ReLU(inplace=True)

        # AVD pool before SplitAttn (avd_first=True)
        self.avd_first: nn.Module = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and avd_first
            else nn.Identity()
        )

        # Split-Attention 3×3 conv (always uses conv_stride=1 when avd active)
        self.conv2 = _SplitAttn(
            group_width,
            group_width,
            kernel_size=3,
            stride=conv_stride,
            padding=dilation,
            dilation=dilation,
            groups=groups,
            radix=radix,
        )
        # bn2 is Identity — SplitAttn already contains bn0 after its conv
        self.bn2: nn.Module = nn.Identity()
        self.act2: nn.Module = nn.Identity()

        # AVD pool after SplitAttn (avd_first=False, default)
        self.avd_last: nn.Module = (
            nn.AvgPool2d(3, avd_stride, padding=1)
            if avd_stride > 0 and not avd_first
            else nn.Identity()
        )

        # 1×1 project
        self.conv3 = nn.Conv2d(group_width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = x

        out = cast(Tensor, self.conv1(x))
        out = cast(Tensor, self.bn1(out))
        out = cast(Tensor, self.act1(out))

        out = cast(Tensor, self.avd_first(out))

        out = cast(Tensor, self.conv2(out))
        out = cast(Tensor, self.bn2(out))
        out = cast(Tensor, self.act2(out))

        out = cast(Tensor, self.avd_last(out))

        out = cast(Tensor, self.conv3(out))
        out = cast(Tensor, self.bn3(out))

        if self.downsample is not None:
            shortcut = cast(Tensor, self.downsample(x))

        out = out + shortcut
        return cast(Tensor, self.act3(out))


# ---------------------------------------------------------------------------
# Downsampling helpers
# ---------------------------------------------------------------------------


def _make_avg_downsample(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    """avg_down=True: AvgPool2d + 1×1 Conv (no stride in conv) + BN."""
    return nn.Sequential(
        nn.AvgPool2d(stride, stride=stride, ceil_mode=True),
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.BatchNorm2d(out_ch),
    )


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------


def _make_layer(
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int,
    radix: int,
    groups: int,
    avd: bool,
    avd_first: bool,
    avg_down: bool,
    dilation: int = 1,
) -> tuple[nn.Sequential, int]:
    """Build one stage of ResNeSt blocks.

    Returns ``(stage_module, out_channels)``.
    """
    expansion = _ResNeStBottleneck.expansion
    out_ch = planes * expansion

    # Downsample for first block
    downsample: nn.Module | None = None
    if stride != 1 or inplanes != out_ch:
        if avg_down:
            downsample = _make_avg_downsample(inplanes, out_ch, stride)
        else:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    blocks: list[nn.Module] = [
        _ResNeStBottleneck(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            radix=radix,
            groups=groups,
            avd=avd,
            avd_first=avd_first,
            is_first=(stride == 1 and inplanes == out_ch),
            dilation=dilation,
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            _ResNeStBottleneck(
                out_ch,
                planes,
                stride=1,
                radix=radix,
                groups=groups,
                avd=avd,
                avd_first=avd_first,
                dilation=dilation,
            )
        )

    return nn.Sequential(*blocks), out_ch


# ---------------------------------------------------------------------------
# Deep stem
# ---------------------------------------------------------------------------


def _make_stem(
    in_channels: int, stem_width: int, deep_stem: bool
) -> tuple[nn.Sequential, nn.BatchNorm2d, int]:
    """Build the ResNeSt stem components to match timm key layout.

    Returns ``(conv1, bn1, stem_out_channels)`` where:
      - ``conv1``: Sequential whose indexed sub-modules match timm positions
      - ``bn1``: final BN (top-level on the model, not inside conv1)
      - ``stem_out_channels``: channels exiting the stem

    timm deep-stem layout (resnest50d)::

        conv1.0  Conv(in→sw, k=3, s=2)
        conv1.1  BN(sw)
        conv1.3  Conv(sw→sw, k=3, s=1)
        conv1.4  BN(sw)
        conv1.6  Conv(sw→2sw, k=3, s=1)
        bn1      BN(2sw)          ← top-level model attribute

    timm flat-stem layout::

        conv1.0  Conv(in→2sw, k=7, s=2)
        bn1      BN(2sw)          ← top-level model attribute
    """
    out = stem_width * 2
    if deep_stem:
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, stem_width, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, stem_width, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_width, out, 3, stride=1, padding=1, bias=False),
        )
    else:
        conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out, 7, stride=2, padding=3, bias=False),
        )
    bn1 = nn.BatchNorm2d(out)
    return conv1, bn1, out


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_resnest_body(config: ResNeStConfig) -> tuple[
    nn.MaxPool2d,  # maxpool
    nn.Sequential,  # layer1
    nn.Sequential,  # layer2
    nn.Sequential,  # layer3
    nn.Sequential,  # layer4
    list[FeatureInfo],
    int,  # out_channels after layer4
]:
    """Build ResNeSt stages (excluding stem) to share between backbone and classifier."""
    stem_out = config.stem_width * 2

    pool = nn.MaxPool2d(3, stride=2, padding=1)

    widths = (64, 128, 256, 512)
    strides = (1, 2, 2, 2)
    layers_cfg = config.layers
    radix = config.radix
    groups = config.groups
    avd = config.avd
    avd_first = config.avd_first
    avg_down = config.avg_down

    cur = stem_out
    fi: list[FeatureInfo] = []

    layer1, cur = _make_layer(
        cur,
        widths[0],
        layers_cfg[0],
        strides[0],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
    )
    fi.append(FeatureInfo(stage=1, num_channels=widths[0] * 4, reduction=4))
    layer2, cur = _make_layer(
        cur,
        widths[1],
        layers_cfg[1],
        strides[1],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
    )
    fi.append(FeatureInfo(stage=2, num_channels=widths[1] * 4, reduction=8))
    layer3, cur = _make_layer(
        cur,
        widths[2],
        layers_cfg[2],
        strides[2],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
    )
    fi.append(FeatureInfo(stage=3, num_channels=widths[2] * 4, reduction=16))
    layer4, cur = _make_layer(
        cur,
        widths[3],
        layers_cfg[3],
        strides[3],
        radix,
        groups,
        avd,
        avd_first,
        avg_down,
    )
    fi.append(FeatureInfo(stage=4, num_channels=widths[3] * 4, reduction=32))

    return pool, layer1, layer2, layer3, layer4, fi, cur


# ---------------------------------------------------------------------------
# ResNeSt backbone  (task="base")
# ---------------------------------------------------------------------------


class ResNeSt(PretrainedModel, BackboneMixin):
    """ResNeSt feature extractor — outputs (B, 2048, H/32, W/32)."""

    config_class: ClassVar[type[ResNeStConfig]] = ResNeStConfig
    base_model_prefix: ClassVar[str] = "resnest"

    def __init__(self, config: ResNeStConfig) -> None:
        super().__init__(config)
        # Stem stored as top-level conv1/bn1 to match timm key layout
        self.conv1, self.bn1, _ = _make_stem(
            config.in_channels, config.stem_width, config.deep_stem
        )
        self.act1 = nn.ReLU(inplace=True)
        pool, l1, l2, l3, l4, fi, _ = _build_resnest_body(config)
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
        x = cast(Tensor, self.conv1(x))
        x = cast(Tensor, self.act1(cast(Tensor, self.bn1(x))))
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
        # Stem stored as top-level conv1/bn1 to match timm key layout
        self.conv1, self.bn1, _ = _make_stem(
            config.in_channels, config.stem_width, config.deep_stem
        )
        self.act1 = nn.ReLU(inplace=True)
        pool, l1, l2, l3, l4, _, out_ch = _build_resnest_body(config)
        self.maxpool = pool
        self.layer1 = l1
        self.layer2 = l2
        self.layer3 = l3
        self.layer4 = l4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(out_ch, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.conv1(x))
        x = cast(Tensor, self.act1(cast(Tensor, self.bn1(x))))
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
