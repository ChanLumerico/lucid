"""CoAtNet backbone and classification head (Dai et al., 2021).

Combines depthwise-separable MBConv stages with multi-head self-attention
Transformer stages. Architecture follows the CoAtNet-0 specification.
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.coatnet._config import CoAtNetConfig

# ---------------------------------------------------------------------------
# MBConv block (Mobile Inverted Bottleneck)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """Mobile Inverted Bottleneck: expand → DWConv → project.

    ``stride`` > 1 applies strided DWConv for spatial downsampling and a
    matching 1×1 shortcut projection.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int = 4,
        stride: int = 1,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand
        self.stride = stride

        self.bn_pre = nn.BatchNorm2d(in_ch)
        self.expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.bn_exp = nn.BatchNorm2d(mid_ch)
        self.dw = nn.Conv2d(
            mid_ch,
            mid_ch,
            3,
            stride=stride,
            padding=1,
            groups=mid_ch,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(mid_ch)
        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)

        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False)
        else:
            self.shortcut = nn.Sequential()  # identity

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = cast(Tensor, self.shortcut(x))

        out = cast(Tensor, self.bn_pre(x))
        exp_out = cast(Tensor, self.bn_exp(cast(Tensor, self.expand(out))))
        out = F.gelu(exp_out)
        dw_out = cast(Tensor, self.bn_dw(cast(Tensor, self.dw(out))))
        out = F.gelu(dw_out)
        out = cast(Tensor, self.project(out))
        return out + shortcut


# ---------------------------------------------------------------------------
# Transformer block used in later stages
# ---------------------------------------------------------------------------


class _TransformerBlock(nn.Module):
    """Standard pre-norm Transformer block: LayerNorm → MHA → residual → FFN."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        n = cast(Tensor, self.norm1(x))
        attn_out, _ = self.attn(n, n, n)
        x = x + attn_out
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Transformer stage wrapper (handles flatten/unflatten around spatial map)
# ---------------------------------------------------------------------------


class _TransformerStage(nn.Module):
    """Wrap a sequence of Transformer blocks applied on a spatial feature map.

    The first block optionally applies 2× average-pooling for downsampling
    (to match MBConv stage strides) before flattening into sequence form.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        num_heads: int,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.pool: nn.Module = nn.MaxPool2d(2) if downsample else nn.Sequential()
        # Channel projection if needed
        self.proj: nn.Module
        if in_ch != out_ch:
            self.proj = nn.Linear(in_ch, out_ch)
        else:
            self.proj = nn.Sequential()
        self.blocks = nn.ModuleList(
            [_TransformerBlock(out_ch, num_heads) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W)
        x = cast(Tensor, self.pool(x))
        B, C, H, W = x.shape
        # Flatten spatial → sequence
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        x = cast(Tensor, self.proj(x))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))
        # Unflatten back to spatial map
        D = x.shape[2]
        x = x.permute(0, 2, 1).reshape(B, D, H, W)
        return x


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body(
    config: CoAtNetConfig,
) -> tuple[
    nn.Sequential,  # stem
    nn.Sequential,  # s0
    nn.Sequential,  # s1
    nn.Sequential,  # s2
    _TransformerStage,  # s3
    _TransformerStage,  # s4
    list[FeatureInfo],
]:
    d = config.dims
    n = config.blocks_per_stage
    exp = config.mbconv_expand
    heads = config.attn_heads

    # Stem: Conv3×3 stride=2 → BN → ReLU
    stem_ch = d[0]
    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.ReLU(inplace=True),
    )

    # S0: MBConv blocks (no striding on blocks; MaxPool at end)
    s0_layers: list[nn.Module] = []
    for i in range(n[0]):
        s0_layers.append(_MBConv(stem_ch, d[0], expand=1, stride=1))
    s0_layers.append(nn.MaxPool2d(2))
    s0 = nn.Sequential(*s0_layers)

    # S1: MBConv blocks, first block has stride=2 to handle downsampling
    s1_layers: list[nn.Module] = []
    s1_layers.append(_MBConv(d[0], d[1], expand=exp, stride=2))
    for _ in range(1, n[1]):
        s1_layers.append(_MBConv(d[1], d[1], expand=exp, stride=1))
    s1 = nn.Sequential(*s1_layers)

    # S2: MBConv blocks, first block stride=2
    s2_layers: list[nn.Module] = []
    s2_layers.append(_MBConv(d[1], d[2], expand=exp, stride=2))
    for _ in range(1, n[2]):
        s2_layers.append(_MBConv(d[2], d[2], expand=exp, stride=1))
    s2 = nn.Sequential(*s2_layers)

    # S3: Transformer stage, first has spatial downsampling (pool)
    s3 = _TransformerStage(d[2], d[3], n[3], heads[0], downsample=True)

    # S4: Transformer stage, no downsampling
    s4 = _TransformerStage(d[3], d[4], n[4], heads[1], downsample=False)

    feature_info = [
        FeatureInfo(stage=1, num_channels=d[0], reduction=4),
        FeatureInfo(stage=2, num_channels=d[1], reduction=8),
        FeatureInfo(stage=3, num_channels=d[2], reduction=16),
        FeatureInfo(stage=4, num_channels=d[3], reduction=32),
        FeatureInfo(stage=5, num_channels=d[4], reduction=32),
    ]
    return stem, s0, s1, s2, s3, s4, feature_info


# ---------------------------------------------------------------------------
# CoAtNet backbone (task="base")
# ---------------------------------------------------------------------------


class CoAtNet(PretrainedModel, BackboneMixin):
    """CoAtNet feature extractor — returns spatial feature map from S4.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` shaped
    ``(B, C, H/32, W/32)``.
    """

    config_class: ClassVar[type[CoAtNetConfig]] = CoAtNetConfig
    base_model_prefix: ClassVar[str] = "coatnet"

    def __init__(self, config: CoAtNetConfig) -> None:
        super().__init__(config)
        stem, s0, s1, s2, s3, s4, fi = _build_body(config)
        self.stem = stem
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self._feature_info = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.s0(x))
        x = cast(Tensor, self.s1(x))
        x = cast(Tensor, self.s2(x))
        x = cast(Tensor, self.s3(x))
        x = cast(Tensor, self.s4(x))
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# CoAtNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class CoAtNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """CoAtNet with global average pooling + linear classification head."""

    config_class: ClassVar[type[CoAtNetConfig]] = CoAtNetConfig
    base_model_prefix: ClassVar[str] = "coatnet"

    def __init__(self, config: CoAtNetConfig) -> None:
        super().__init__(config)
        stem, s0, s1, s2, s3, s4, _ = _build_body(config)
        self.stem = stem
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self._build_classifier(
            config.dims[-1], config.num_classes, dropout=config.dropout
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.s0(x))
        x = cast(Tensor, self.s1(x))
        x = cast(Tensor, self.s2(x))
        x = cast(Tensor, self.s3(x))
        x = cast(Tensor, self.s4(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
