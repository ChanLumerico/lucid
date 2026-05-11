"""CSPNet backbone and classification head (Wang et al., 2019).

Cross Stage Partial Networks split the feature map into two branches,
apply dense/residual blocks to one branch, then concatenate and project.
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.cspnet._config import CSPNetConfig

# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class _ConvBnReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU helper."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(
            Tensor, self.relu(cast(Tensor, self.bn(cast(Tensor, self.conv(x)))))
        )


class _ResBottleneck(nn.Module):
    """Standard ResNet bottleneck: 1×1 → 3×3 → 1×1 (expansion=4).

    A downsample projection is added automatically when channels differ.
    """

    expansion: int = 4

    def __init__(
        self,
        in_ch: int,
        mid_ch: int,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_ch)
        self.conv3 = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

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
        return cast(Tensor, self.relu(out + identity))


# ---------------------------------------------------------------------------
# CSP bottleneck stage
# ---------------------------------------------------------------------------


class _CSPBottleneck(nn.Module):
    """One CSP stage.

    Channels are halved into two branches:
      • Branch A: single 1×1 conv (identity-like pass-through)
      • Branch B: N × ResBottleneck blocks
    Both are concatenated and projected back to ``out_ch``.
    """

    def __init__(self, in_ch: int, out_ch: int, n: int) -> None:
        super().__init__()
        mid_ch = out_ch // 2
        exp = _ResBottleneck.expansion

        # Branch A: 1×1 projection
        self.branch_a = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn_a = nn.BatchNorm2d(mid_ch)

        # Branch B: N bottleneck blocks
        # First block: handle channel mismatch with a downsample projection.
        downsample: nn.Module | None = None
        if in_ch != mid_ch * exp:
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, mid_ch * exp, 1, bias=False),
                nn.BatchNorm2d(mid_ch * exp),
            )

        blocks: list[nn.Module] = [_ResBottleneck(in_ch, mid_ch, downsample)]
        for _ in range(1, n):
            blocks.append(_ResBottleneck(mid_ch * exp, mid_ch))
        self.branch_b = nn.Sequential(*blocks)

        # Transition: concat(branch_a, branch_b) → out_ch
        self.transition = nn.Sequential(
            nn.Conv2d(mid_ch + mid_ch * exp, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        import lucid

        a = cast(
            Tensor, self.relu(cast(Tensor, self.bn_a(cast(Tensor, self.branch_a(x)))))
        )
        b = cast(Tensor, self.branch_b(x))
        concat = lucid.cat([a, b], dim=1)
        return cast(Tensor, self.transition(concat))


# ---------------------------------------------------------------------------
# Stem + body builder
# ---------------------------------------------------------------------------


def _build_body(
    config: CSPNetConfig,
) -> tuple[
    nn.Sequential,  # stem
    nn.MaxPool2d,  # pool
    list[_CSPBottleneck],  # 4 CSP stages
    list[nn.Sequential],  # 4 downsampling convs (between stages)
    list[FeatureInfo],
]:
    ch = config.channels
    layers = config.layers

    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, ch[0], 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(ch[0]),
        nn.ReLU(inplace=True),
    )
    pool = nn.MaxPool2d(3, stride=2, padding=1)

    stages: list[_CSPBottleneck] = []
    downs: list[nn.Sequential] = []

    in_ch = ch[0]
    for i, (out_ch, n) in enumerate(zip(ch, layers)):
        if i > 0:
            # 2× spatial downsampling before each stage (except first)
            down = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
            downs.append(down)
            in_ch = out_ch
        else:
            # Dummy entry so indexing is consistent (stage 0 has no pre-down)
            downs.append(nn.Sequential())

        stage = _CSPBottleneck(in_ch, out_ch, n)
        stages.append(stage)
        in_ch = out_ch

    reduction_factors = [4, 8, 16, 32]
    feature_info = [
        FeatureInfo(stage=i + 1, num_channels=ch[i], reduction=reduction_factors[i])
        for i in range(4)
    ]
    return stem, pool, stages, downs, feature_info


# ---------------------------------------------------------------------------
# CSPNet backbone (task="base")
# ---------------------------------------------------------------------------


class CSPNet(PretrainedModel, BackboneMixin):
    """CSPResNet feature extractor — no classification head.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` shaped
    ``(B, C, H/32, W/32)`` from stage 4.
    """

    config_class: ClassVar[type[CSPNetConfig]] = CSPNetConfig
    base_model_prefix: ClassVar[str] = "cspnet"

    def __init__(self, config: CSPNetConfig) -> None:
        super().__init__(config)
        stem, pool, stages, downs, fi = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.stage0 = stages[0]
        self.stage1 = stages[1]
        self.stage2 = stages[2]
        self.stage3 = stages[3]
        # downs[0] is a no-op Sequential (stage 0 has no pre-downsampling)
        self.down1 = downs[1]
        self.down2 = downs[2]
        self.down3 = downs[3]
        self._feature_info = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.maxpool(x))
        x = cast(Tensor, self.stage0(x))
        x = cast(Tensor, self.down1(x))
        x = cast(Tensor, self.stage1(x))
        x = cast(Tensor, self.down2(x))
        x = cast(Tensor, self.stage2(x))
        x = cast(Tensor, self.down3(x))
        x = cast(Tensor, self.stage3(x))
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# CSPNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class CSPNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """CSPResNet with global average pooling + linear classification head."""

    config_class: ClassVar[type[CSPNetConfig]] = CSPNetConfig
    base_model_prefix: ClassVar[str] = "cspnet"

    def __init__(self, config: CSPNetConfig) -> None:
        super().__init__(config)
        stem, pool, stages, downs, _ = _build_body(config)
        self.stem = stem
        self.maxpool = pool
        self.stage0 = stages[0]
        self.stage1 = stages[1]
        self.stage2 = stages[2]
        self.stage3 = stages[3]
        self.down1 = downs[1]
        self.down2 = downs[2]
        self.down3 = downs[3]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(config.channels[-1], config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.maxpool(x))
        x = cast(Tensor, self.stage0(x))
        x = cast(Tensor, self.down1(x))
        x = cast(Tensor, self.stage1(x))
        x = cast(Tensor, self.down2(x))
        x = cast(Tensor, self.stage2(x))
        x = cast(Tensor, self.down3(x))
        x = cast(Tensor, self.stage3(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
