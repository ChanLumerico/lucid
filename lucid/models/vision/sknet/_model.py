"""SKNet backbone and classification head (Li et al., 2019).

Paper: "Selective Kernel Networks"
This implementation follows the timm ``skresnet50`` architecture:
SelectiveKernel with ``split_input=True`` (each branch receives half the
input channels), two 3×3 branches (second with dilation=2 to mimic 5×5),
and a Conv2d-based attention module.

For SK-ResNeXt, ``cardinality`` and ``base_width`` together determine the
bottleneck width per the ResNeXt formula:
  width = int(planes * (base_width / 64)) * cardinality
"""

import math
from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._common import make_divisible as _make_divisible
from lucid.models.vision.sknet._config import SKNetConfig

# ---------------------------------------------------------------------------
# SelectiveKernelAttn — attention module (Conv2d-based, as in timm)
# ---------------------------------------------------------------------------


class _SelectiveKernelAttn(nn.Module):
    """Attention module for SelectiveKernel.

    Input shape: ``(B, num_paths, C, H, W)`` — stacked branch outputs.
    Performs global-average-pool over the element-wise sum, reduces
    channels, then outputs per-path softmax weights of shape
    ``(B, num_paths, C, 1, 1)``.
    """

    def __init__(
        self,
        channels: int,
        num_paths: int = 2,
        attn_channels: int = 32,
    ) -> None:
        super().__init__()
        self.num_paths = num_paths
        # Conv2d 1×1 — same param count as Linear but keeps (B, C, 1, 1) layout
        self.fc_reduce = nn.Conv2d(channels, attn_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(attn_channels)
        self.act = nn.ReLU(inplace=True)
        self.fc_select = nn.Conv2d(
            attn_channels, channels * num_paths, kernel_size=1, bias=False
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, num_paths, C, H, W)
        # Sum paths then global-average-pool → (B, C, 1, 1)
        b = x.shape[0]
        c = x.shape[2]

        # Sum over path dimension (dim=1)
        summed = x[:, 0, :, :, :]
        for p in range(1, self.num_paths):
            summed = summed + x[:, p, :, :, :]

        # Global average pool: mean over H and W
        gap = cast(Tensor, nn.AdaptiveAvgPool2d(1)(summed))  # (B, C, 1, 1)

        z = cast(Tensor, self.fc_reduce(gap))
        z = cast(Tensor, self.bn(z))
        z = cast(Tensor, self.act(z))
        z = cast(Tensor, self.fc_select(z))  # (B, C*num_paths, 1, 1)

        # Reshape to (B, num_paths, C, 1, 1) then softmax over path dim
        z = z.reshape(b, self.num_paths, c, 1, 1)
        z = F.softmax(z, dim=1)
        return z


# ---------------------------------------------------------------------------
# ConvBnAct — simple conv + BN + ReLU block
# ---------------------------------------------------------------------------


class _ConvBnAct(nn.Module):
    """Conv2d → BatchNorm2d → ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        apply_act: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.apply_act = apply_act
        if apply_act:
            self.act: nn.Module = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        out = cast(Tensor, self.conv(x))
        out = cast(Tensor, self.bn(out))
        if self.apply_act:
            out = cast(Tensor, self.act(out))
        return out


# ---------------------------------------------------------------------------
# SelectiveKernel — multi-branch conv with split input
# ---------------------------------------------------------------------------


class _SelectiveKernel(nn.Module):
    """Two-branch selective-kernel convolution.

    By default ``split_input=True``: the input is split in half along the
    channel axis and each half is fed to one branch.  This keeps the param
    count comparable to a single grouped 3×3 conv.

    Branches are 3×3 with dilation 1 and 3×3 with dilation 2 (mimicking 5×5).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        split_input: bool = True,
        rd_ratio: float = 1.0 / 16,
        rd_divisor: int = 8,
    ) -> None:
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        # keep_3x3=True: kernels [3, 5] → 3×3 with dilations [1, 2]
        dilations = [dilation, dilation * 2]
        kernel_sizes = [3, 3]
        paddings = [d for d in dilations]

        self.num_paths = 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.split_input = split_input

        path_in = in_channels // self.num_paths if split_input else in_channels
        effective_groups = min(out_channels, groups)

        self.paths = nn.ModuleList(
            [
                _ConvBnAct(
                    path_in,
                    out_channels,
                    kernel_size=k,
                    stride=stride,
                    padding=p,
                    dilation=d,
                    groups=effective_groups,
                )
                for k, d, p in zip(kernel_sizes, dilations, paddings)
            ]
        )

        attn_channels = _make_divisible(out_channels * rd_ratio, divisor=rd_divisor)
        self.attn = _SelectiveKernelAttn(out_channels, self.num_paths, attn_channels)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        if self.split_input:
            half = self.in_channels // self.num_paths
            x_paths = [
                cast(Tensor, self.paths[i](x[:, i * half : (i + 1) * half, :, :]))
                for i in range(self.num_paths)
            ]
        else:
            x_paths = [cast(Tensor, op(x)) for op in self.paths]

        # Stack: (B, num_paths, C, H, W)
        stacked = lucid.stack(x_paths, dim=1)
        # Attention weights: (B, num_paths, C, 1, 1)
        attn = cast(Tensor, self.attn(stacked))
        # Weighted sum
        weighted = stacked * attn
        out = weighted[:, 0, :, :, :]
        for p in range(1, self.num_paths):
            out = out + weighted[:, p, :, :, :]
        return out


# ---------------------------------------------------------------------------
# SelectiveKernelBasic (expansion=1, for SK-ResNet-18/34)
# ---------------------------------------------------------------------------


class _SelectiveKernelBasic(nn.Module):
    """SK 3×3 → SK 3×3 basic block (expansion=1) for shallow SK-ResNets.

    Both 3×3 convolutions are replaced by SelectiveKernel units (two-branch
    parallel 3×3 convolutions with channel attention), giving a full SK
    treatment of the ResNet-18/34 basic block.
    """

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        cardinality: int = 1,
        base_width: int = 64,
        split_input: bool = False,
        rd_ratio: float = 1.0 / 16,
        rd_divisor: int = 8,
    ) -> None:
        super().__init__()
        width = int(math.floor(planes * (base_width / 64)) * cardinality)

        self.conv1 = _SelectiveKernel(
            inplanes,
            width,
            stride=stride,
            groups=cardinality,
            split_input=split_input,
            rd_ratio=rd_ratio,
            rd_divisor=rd_divisor,
        )
        self.conv2 = _SelectiveKernel(
            width,
            planes * self.expansion,
            groups=cardinality,
            split_input=split_input,
            rd_ratio=rd_ratio,
            rd_divisor=rd_divisor,
        )
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x

        out = cast(Tensor, self.conv1(x))
        out = cast(Tensor, self.conv2(out))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        out = out + identity
        return cast(Tensor, self.act(out))


# ---------------------------------------------------------------------------
# SelectiveKernelBottleneck
# ---------------------------------------------------------------------------


class _SelectiveKernelBottleneck(nn.Module):
    """1×1 → SK(3×3/3×3 dilated) → 1×1 bottleneck with SK attention."""

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        cardinality: int = 1,
        base_width: int = 64,
        split_input: bool = True,
        rd_ratio: float = 1.0 / 16,
        rd_divisor: int = 8,
    ) -> None:
        super().__init__()
        # ResNeXt-style width computation (planes for standard ResNet when base_width=64)
        width = int(math.floor(planes * (base_width / 64)) * cardinality)
        outplanes = planes * self.expansion

        self.conv1 = _ConvBnAct(inplanes, width, kernel_size=1, padding=0)
        self.conv2 = _SelectiveKernel(
            width,
            width,
            stride=stride,
            groups=cardinality,
            split_input=split_input,
            rd_ratio=rd_ratio,
            rd_divisor=rd_divisor,
        )
        self.conv3 = _ConvBnAct(
            width, outplanes, kernel_size=1, padding=0, apply_act=False
        )
        self.act = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        identity = x

        out = cast(Tensor, self.conv1(x))
        out = cast(Tensor, self.conv2(out))
        out = cast(Tensor, self.conv3(out))

        if self.downsample is not None:
            identity = cast(Tensor, self.downsample(x))

        out = out + identity
        return cast(Tensor, self.act(out))


# ---------------------------------------------------------------------------
# Stage builder
# ---------------------------------------------------------------------------


def _make_stage(
    inplanes: int,
    planes: int,
    num_blocks: int,
    stride: int,
    cardinality: int,
    base_width: int,
    split_input: bool,
    rd_ratio: float,
    rd_divisor: int,
    block_type: str = "bottleneck",
) -> tuple[nn.Sequential, int]:
    """Build one ResNet stage of SK blocks (bottleneck or basic)."""
    block_cls: type[_SelectiveKernelBottleneck] | type[_SelectiveKernelBasic]
    if block_type == "basic":
        block_cls = _SelectiveKernelBasic
    else:
        block_cls = _SelectiveKernelBottleneck

    outplanes = planes * block_cls.expansion

    downsample: nn.Module | None = None
    if stride != 1 or inplanes != outplanes:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(outplanes),
        )

    block_kwargs = dict(
        cardinality=cardinality,
        base_width=base_width,
        split_input=split_input,
        rd_ratio=rd_ratio,
        rd_divisor=rd_divisor,
    )

    blocks: list[nn.Module] = [
        block_cls(
            inplanes,
            planes,
            stride=stride,
            downsample=downsample,
            **block_kwargs,  # type: ignore[arg-type]
        )
    ]
    for _ in range(1, num_blocks):
        blocks.append(
            block_cls(
                outplanes,
                planes,
                **block_kwargs,  # type: ignore[arg-type]
            )
        )

    return nn.Sequential(*blocks), outplanes


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
    """Build all layers from config; return them plus the stage feature-info list."""
    stem = nn.Sequential(
        nn.Conv2d(
            config.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        ),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    planes_per_stage = [64, 128, 256, 512]
    strides = [1, 2, 2, 2]

    kw = dict(
        cardinality=config.cardinality,
        base_width=config.base_width,
        split_input=config.split_input,
        rd_ratio=config.rd_ratio,
        rd_divisor=config.rd_divisor,
        block_type=config.block_type,
    )

    cur = 64
    layers: list[nn.Sequential] = []
    feature_info: list[FeatureInfo] = []
    reductions = [4, 8, 16, 32]

    for stage_idx, (planes, stride) in enumerate(zip(planes_per_stage, strides)):
        layer, cur = _make_stage(
            cur,
            planes,
            config.layers[stage_idx],
            stride,
            **kw,  # type: ignore[arg-type]
        )
        layers.append(layer)
        feature_info.append(
            FeatureInfo(
                stage=stage_idx + 1, num_channels=cur, reduction=reductions[stage_idx]
            )
        )

    return stem, pool, layers[0], layers[1], layers[2], layers[3], feature_info


# ---------------------------------------------------------------------------
# SKNet backbone (task="base")
# ---------------------------------------------------------------------------


class SKNet(PretrainedModel, BackboneMixin):
    """SK-ResNet feature extractor — no classification head.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` of shape
    ``(B, 2048, H/32, W/32)`` for typical inputs.
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

        expansion = (
            _SelectiveKernelBasic.expansion
            if config.block_type == "basic"
            else _SelectiveKernelBottleneck.expansion
        )
        final_channels = 512 * expansion  # 512 (basic) or 2048 (bottleneck)
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
