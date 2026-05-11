"""NFNet backbone and classifier (Brock et al., 2021).

Paper: "High-Performance Large-Scale Image Recognition Without Normalization"

Key ideas:
  - BatchNorm removed entirely — Scaled Weight Standardization replaces it.
  - ScaledStdConv2d: normalise each conv filter to zero mean + unit variance
    over its fan-in, then scale by a learned per-filter gain and 1/sqrt(fan_in).
  - NFBlock: four ScaledStdConv2d layers (1×1, 3×3, 3×3, 1×1) + SE gating.
    Expected signal variance tracks through alpha/beta bookkeeping.
  - Stem: four ScaledStdConv2d layers (stride-2, then stride-2 again).
  - Head: AdaptiveAvgPool → Dropout → Linear.

Architecture (NFNet-F0, widths=(256,512,1536,1536), depths=(1,2,6,3)):
  Stem     : ScaledStdConv2d ×4, output channels = widths[0] // 2
  Stage 0  : 1 × NFBlock(stem_ch → 256),   stride=1
  Stage 1  : 2 × NFBlock(256   → 512),    stride=2
  Stage 2  : 6 × NFBlock(512   → 1536),   stride=2
  Stage 3  : 3 × NFBlock(1536  → 1536),   stride=2
  Head     : AdaptiveAvgPool(1×1) → Dropout → Linear(1536, num_classes)
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
from lucid.models.vision.nfnet._config import NFNetConfig


# ---------------------------------------------------------------------------
# Scaled Weight Standardization convolution
# ---------------------------------------------------------------------------


class _ScaledStdConv2d(nn.Module):
    """Conv2d with Scaled Weight Standardization.

    Each output-filter's weights are normalised to zero mean and unit variance
    over the fan-in (in_ch * kH * kW / groups) slice, then scaled by
    ``gain * fan_in^{-0.5}`` so the output variance is ≈ 1.0 at initialisation.

    Reference: Brock et al. (2021) §3, Eq. 1–3.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = True,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel, stride=stride, padding=padding, groups=groups, bias=bias
        )
        # Per-filter learnable gain, initialised to 1.0
        self.gain = nn.Parameter(lucid.ones(out_ch, 1, 1, 1))
        self.eps = eps
        fan_in = in_ch * kernel * kernel // groups
        # Precomputed scale = 1 / sqrt(fan_in)
        self._scale: float = fan_in ** -0.5

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        w = self.conv.weight  # (out_ch, in_ch//groups, kH, kW)
        # Flatten fan-in dims: (out_ch, fan_in)
        w_flat = w.reshape(w.shape[0], -1)
        # Per-filter mean: (out_ch, 1)
        w_mean = w_flat.mean(dim=1, keepdim=True)
        w_centered = w_flat - w_mean
        # Per-filter variance → std
        w_var = (w_centered * w_centered).mean(dim=1, keepdim=True)
        w_std = (w_var + self.eps) ** 0.5
        w_normed = w_centered / w_std  # (out_ch, fan_in)
        # Reshape back and apply gain + fan-in scaling
        w_final = w_normed.reshape(w.shape) * (self.gain * self._scale)
        return F.conv2d(
            x,
            w_final,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            groups=self.conv.groups,
        )


# ---------------------------------------------------------------------------
# Squeeze-Excitation block (sigmoid gating)
# ---------------------------------------------------------------------------


class _SEBlock(nn.Module):
    """Channel squeeze-and-excitation with sigmoid gating.

    Applied on the branch output (before scaling by alpha).  The output is
    multiplied by 2 so the gating is centred at 1.0 when sigmoid ≈ 0.5.
    """

    def __init__(self, ch: int, se_ch: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch, se_ch)
        self.fc2 = nn.Linear(se_ch, ch)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W)
        s = cast(Tensor, self.avgpool(x)).flatten(1)  # (B, C)
        s = cast(Tensor, self.act(cast(Tensor, self.fc1(s))))
        s = F.sigmoid(cast(Tensor, self.fc2(s)))
        # Multiply by 2 so gate is centred at identity
        return x * s.reshape(s.shape[0], s.shape[1], 1, 1) * 2.0


# ---------------------------------------------------------------------------
# NFBlock: the core normalizer-free residual block
# ---------------------------------------------------------------------------


class _NFBlock(nn.Module):
    """Normalizer-Free residual block.

    Structure (pre-activation style, no BatchNorm):

      out = act(x) / beta            # normalise by expected std
      out = conv1×1 → act
      out = conv3×3 (stride, grouped) → act
      out = conv3×3 (grouped) → act  # extra 3×3
      out = conv1×1
      out = SE(out)
      out = out * alpha * skip_gain  # branch scale; skip_gain init=0
      x   = shortcut(x) if shape differs else x
      return x + out

    ``beta``  — expected std of the input (tracks across blocks per stage).
    ``alpha`` — per-block scale applied to the residual branch.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        groups: int = 1,
        alpha: float = 0.2,
        beta: float = 1.0,
        se_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.alpha = alpha
        self.act = nn.GELU()

        # Main branch — four ScaledStdConv2d layers
        self.conv0 = _ScaledStdConv2d(in_ch, out_ch, 1)
        self.conv1 = _ScaledStdConv2d(
            out_ch, out_ch, 3, stride=stride, padding=1, groups=groups
        )
        self.conv1b = _ScaledStdConv2d(out_ch, out_ch, 3, padding=1, groups=groups)
        self.conv2 = _ScaledStdConv2d(out_ch, out_ch, 1)

        # Squeeze-Excitation applied to out_ch
        se_ch = max(1, int(in_ch * se_ratio))
        self.se = _SEBlock(out_ch, se_ch)

        # Skip connection (1×1 ScaledStdConv if shapes differ)
        self.downsample: _ScaledStdConv2d | None = (
            _ScaledStdConv2d(in_ch, out_ch, 1, stride=stride, bias=False)
            if stride > 1 or in_ch != out_ch
            else None
        )

        # Learnable branch scale (initialised to 0 → starts as identity)
        self.skip_gain = nn.Parameter(lucid.zeros(1))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Normalise by expected std before entering the branch
        out = cast(Tensor, self.act(x)) * (1.0 / self.beta)

        out = cast(Tensor, self.act(cast(Tensor, self.conv0(out))))
        out = cast(Tensor, self.act(cast(Tensor, self.conv1(out))))
        out = cast(Tensor, self.act(cast(Tensor, self.conv1b(out))))
        out = cast(Tensor, self.conv2(out))
        out = cast(Tensor, self.se(out))
        # Scale residual branch by alpha * skip_gain
        out = out * (self.alpha * self.skip_gain)

        skip = x if self.downsample is None else cast(Tensor, self.downsample(x))
        return out + skip


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_nfnet(
    cfg: NFNetConfig,
) -> tuple[
    nn.Sequential,          # stem
    nn.ModuleList,          # stages
    nn.AdaptiveAvgPool2d,   # pool
    list[FeatureInfo],
    int,                    # final channels
]:
    """Build NFNet stem + stages from a config."""
    # Stem: four ScaledStdConv2d layers (two stride-2)
    stem_ch = cfg.widths[0] // 2
    stem = nn.Sequential(
        _ScaledStdConv2d(cfg.in_channels, 16, 3, stride=2, padding=1),
        nn.GELU(),
        _ScaledStdConv2d(16, 32, 3, padding=1),
        nn.GELU(),
        _ScaledStdConv2d(32, 64, 3, padding=1),
        nn.GELU(),
        _ScaledStdConv2d(64, stem_ch, 3, stride=2, padding=1),
        nn.GELU(),
    )

    stages: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4  # stem reduces by ×4 (stride-2 × stride-2)

    in_ch = stem_ch
    for stage_idx, (width, depth) in enumerate(zip(cfg.widths, cfg.depths)):
        out_ch = width
        groups = max(1, out_ch // cfg.group_size)
        # First block may stride=2 (except stage 0)
        stride = 2 if stage_idx > 0 else 1
        if stage_idx > 0:
            reduction *= 2

        # Variance bookkeeping across blocks in this stage.
        # Expected variance grows as 1 + alpha^2 each block; beta = sqrt(expected var).
        expected_var = 1.0
        blocks: list[nn.Module] = []
        for block_idx in range(depth):
            beta = math.sqrt(expected_var)
            s = stride if block_idx == 0 else 1
            blocks.append(
                _NFBlock(
                    in_ch=in_ch if block_idx == 0 else out_ch,
                    out_ch=out_ch,
                    stride=s,
                    groups=groups,
                    alpha=cfg.alpha,
                    beta=beta,
                    se_ratio=cfg.se_ratio,
                )
            )
            expected_var += cfg.alpha ** 2
            in_ch = out_ch

        stages.append(nn.Sequential(*blocks))
        fi.append(FeatureInfo(stage=stage_idx + 1, num_channels=out_ch, reduction=reduction))

    pool = nn.AdaptiveAvgPool2d(1)
    return stem, nn.ModuleList(stages), pool, fi, in_ch


# ---------------------------------------------------------------------------
# NFNet backbone
# ---------------------------------------------------------------------------


class NFNet(PretrainedModel, BackboneMixin):
    """NFNet feature extractor — global-average-pooled features from the final stage.

    ``forward_features`` returns ``(B, C)`` where C = widths[-1].
    ``forward`` wraps it in ``BaseModelOutput``.
    """

    config_class: ClassVar[type[NFNetConfig]] = NFNetConfig
    base_model_prefix: ClassVar[str] = "nfnet"

    def __init__(self, config: NFNetConfig) -> None:
        super().__init__(config)
        stem, stages, pool, fi, out_dim = _build_nfnet(config)
        self.stem = stem
        self.stages = stages
        self.pool = pool
        self._feature_info = fi
        self._out_dim = out_dim
        self.act = nn.GELU()

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        x = cast(Tensor, self.act(x))
        x = cast(Tensor, self.pool(x)).flatten(1)  # (B, C)
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# NFNet for image classification
# ---------------------------------------------------------------------------


class NFNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """NFNet with AdaptiveAvgPool + Dropout + Linear classifier."""

    config_class: ClassVar[type[NFNetConfig]] = NFNetConfig
    base_model_prefix: ClassVar[str] = "nfnet"

    def __init__(self, config: NFNetConfig) -> None:
        super().__init__(config)
        stem, stages, pool, _, out_dim = _build_nfnet(config)
        self.stem = stem
        self.stages = stages
        self.pool = pool
        self.act = nn.GELU()
        self._build_classifier(out_dim, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        x = cast(Tensor, self.act(x))
        x = cast(Tensor, self.pool(x)).flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
