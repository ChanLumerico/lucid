"""InceptionNeXt backbone and classifier (Yu et al., 2023).

Paper: "InceptionNeXt: When Inception Meets ConvNeXt"

Key ideas:
  1. ConvNeXt's large 7×7 DWConv is decomposed into parallel branches
     (Inception-style) to reduce computation while maintaining receptive field.
  2. Four branches on channel splits:
       - Identity (dim//4 channels, no-op)
       - 3×3 DWConv (dim//4 channels)
       - 1×K + K×1 band DWConv (dim//4 channels, K=11)
       - 3×3 DWConv for remaining channels (high-frequency branch)
  3. Concatenation of branches replaces the single DWConv in ConvNeXt block.
  4. Same patchify stem, LN-based downsampling, and LayerScale as ConvNeXt.

Architecture (InceptionNeXt-T, dims=(96,192,384,768)):
  Stem     : Conv2d(4×4, stride=4) → LN               → (56×56, 96)
  Stage 1  : 3 × InceptionNeXtBlock(96)  → LN-Down(2×) → (28×28, 192)
  Stage 2  : 3 × InceptionNeXtBlock(192) → LN-Down(2×) → (14×14, 384)
  Stage 3  : 9 × InceptionNeXtBlock(384) → LN-Down(2×) → (7×7,  768)
  Stage 4  : 3 × InceptionNeXtBlock(768)               → (7×7,  768)
  Head     : AdaptiveAvgPool(1×1) → LN → FC
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.inception_next._config import InceptionNeXtConfig

# ---------------------------------------------------------------------------
# InceptionDWConv2d — multi-branch depthwise conv
# ---------------------------------------------------------------------------


class _InceptionDWConv2d(nn.Module):
    """Decomposed DWConv with 4 parallel branches operating on channel splits.

    Matches timm's ``InceptionDWConv2d`` with ``branch_ratio=0.125``:
      gc = int(dim * 0.125)  (channels per small branch)
      branch 0: identity, dim - 3*gc channels (majority passthrough)
      branch 1: 3×3 DWConv on gc channels
      branch 2: 1×K DWConv → K×1 DWConv on gc channels (band conv)
      branch 3: 3×3 DWConv on gc channels (high-frequency)
    All outputs concatenated to recover ``dim`` channels.
    """

    def __init__(
        self, dim: int, band_kernel: int = 11, branch_ratio: float = 0.125
    ) -> None:
        super().__init__()
        gc = int(dim * branch_ratio)  # channels per small branch
        self.gc = gc
        self.identity_chs = dim - 3 * gc  # majority passthrough

        # branch 1: 3×3 DWConv
        self.dw3x3 = nn.Conv2d(gc, gc, 3, padding=1, groups=gc)

        # branch 2: 1×K → K×1 (sequential band conv)
        pad = band_kernel // 2
        self.dw_h = nn.Conv2d(gc, gc, (1, band_kernel), padding=(0, pad), groups=gc)
        self.dw_v = nn.Conv2d(gc, gc, (band_kernel, 1), padding=(pad, 0), groups=gc)

        # branch 3: 3×3 DWConv on gc channels
        self.dw3x3_b = nn.Conv2d(gc, gc, 3, padding=1, groups=gc)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        id_chs = self.identity_chs
        gc = self.gc
        # Split along channel dim: identity | dw3x3 | band | dw3x3_b
        x0 = x[:, :id_chs, :, :]  # identity passthrough
        x1 = x[:, id_chs : id_chs + gc, :, :]  # 3×3 DWConv
        x2 = x[:, id_chs + gc : id_chs + 2 * gc, :, :]  # band conv
        x3 = x[:, id_chs + 2 * gc :, :, :]  # high-freq 3×3

        y0 = x0
        y1 = cast(Tensor, self.dw3x3(x1))
        y2 = cast(Tensor, self.dw_v(cast(Tensor, self.dw_h(x2))))
        y3 = cast(Tensor, self.dw3x3_b(x3))

        return lucid.cat([y0, y1, y2, y3], dim=1)


# ---------------------------------------------------------------------------
# InceptionNeXt block (ConvNeXt block with InceptionDWConv2d)
# ---------------------------------------------------------------------------


class _InceptionNeXtBlock(nn.Module):
    """ConvNeXt-style block with InceptionDWConv2d replacing the 7×7 DWConv."""

    def __init__(
        self, dim: int, band_kernel: int, layer_scale_init: float = 1e-6
    ) -> None:
        super().__init__()
        self.dwconv = _InceptionDWConv2d(dim, band_kernel)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(lucid.full((dim,), layer_scale_init))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = x
        x = cast(Tensor, self.dwconv(x))  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = cast(Tensor, self.norm(x))
        x = F.gelu(cast(Tensor, self.fc1(x)))
        x = cast(Tensor, self.fc2(x))
        x = x * self.gamma  # layer scale
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return shortcut + x


# ---------------------------------------------------------------------------
# Downsampling (same as ConvNeXt: LN + stride-2 conv)
# ---------------------------------------------------------------------------


class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.conv = nn.Conv2d(in_dim, out_dim, 2, stride=2)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x.permute(0, 2, 3, 1)
        x = cast(Tensor, self.norm(x))
        x = x.permute(0, 3, 1, 2)
        return cast(Tensor, self.conv(x))


# ---------------------------------------------------------------------------
# Stem + norm wrapper (same as ConvNeXt)
# ---------------------------------------------------------------------------


class _StemWithNorm(nn.Module):
    def __init__(self, dim: int, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, dim, 4, stride=4)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.conv(x))
        x = x.permute(0, 2, 3, 1)
        x = cast(Tensor, self.norm(x))
        return x.permute(0, 3, 1, 2)


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_inception_next(cfg: InceptionNeXtConfig) -> tuple[
    _StemWithNorm,
    nn.ModuleList,
    nn.ModuleList,
    nn.LayerNorm,
    list[FeatureInfo],
    int,
]:
    stem = _StemWithNorm(cfg.dims[0], cfg.in_channels)

    stages: list[nn.Module] = []
    downsamplers: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4

    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
        stage = nn.Sequential(
            *[_InceptionNeXtBlock(dim, cfg.band_kernel) for _ in range(depth)]
        )
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < len(cfg.depths) - 1:
            next_dim = cfg.dims[i + 1]
            downsamplers.append(_Downsample(dim, next_dim))
            reduction *= 2

    head_norm = nn.LayerNorm(cfg.dims[-1])
    return (
        stem,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        head_norm,
        fi,
        cfg.dims[-1],
    )


# ---------------------------------------------------------------------------
# InceptionNeXt backbone
# ---------------------------------------------------------------------------


class InceptionNeXt(PretrainedModel, BackboneMixin):
    """InceptionNeXt feature extractor — global avg-pooled final-stage features."""

    config_class: ClassVar[type[InceptionNeXtConfig]] = InceptionNeXtConfig
    base_model_prefix: ClassVar[str] = "inception_next"

    def __init__(self, config: InceptionNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, fi, out_dim = _build_inception_next(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._feature_info = fi
        self._out_dim = out_dim
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        for i, stage in enumerate(self.stages):
            x = cast(Tensor, stage(x))
            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))
        x = cast(Tensor, self.avgpool(x)).flatten(1)
        # head norm in channel-last
        x = x.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 3, 1)
        x = cast(Tensor, self.head_norm(x))
        return x.flatten(1)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# InceptionNeXt for image classification
# ---------------------------------------------------------------------------


class InceptionNeXtForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """InceptionNeXt with AdaptiveAvgPool + LN + FC classifier."""

    config_class: ClassVar[type[InceptionNeXtConfig]] = InceptionNeXtConfig
    base_model_prefix: ClassVar[str] = "inception_next"

    def __init__(self, config: InceptionNeXtConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, _, out_dim = _build_inception_next(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        for i, stage in enumerate(self.stages):
            x = cast(Tensor, stage(x))
            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))
        x = cast(Tensor, self.avgpool(x)).flatten(1)
        x = x.unsqueeze(-1).unsqueeze(-1).permute(0, 2, 3, 1)
        x = cast(Tensor, self.head_norm(x))
        x = x.flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
