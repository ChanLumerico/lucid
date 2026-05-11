"""MaxViT backbone and classifier (Tu et al., 2022).

Paper: "MaxViT: Multi-Axis Vision Transformer"
        https://arxiv.org/abs/2204.01697

Key ideas:
  1. Multi-Axis attention: combine local window attention (block-attention) and
     global grid attention in every block — O(n) overall complexity.
  2. MBConv (mobile inverted bottleneck) in each block for local features.
  3. Window (block) attention: partition spatial into non-overlapping ws×ws
     windows, run MHA within each window.
  4. Grid attention: dilated / strided partition (every ws-th pixel in each
     direction forms a virtual "grid"), run MHA within each grid.

Architecture (MaxViT-T, image=224, ws=7):
  Stem   : Conv3×3(s=2, → 32) → BN → GELU → Conv3×3(s=1, 32→64) → (112×112)
  Stage 1: 2 × MaxViTBlock(64)  → Downsample(stride=2) → (56×56, 128)
  Stage 2: 2 × MaxViTBlock(128) → Downsample(stride=2) → (28×28, 256)
  Stage 3: 5 × MaxViTBlock(256) → Downsample(stride=2) → (14×14, 512)
  Stage 4: 2 × MaxViTBlock(512)
  Head   : AdaptiveAvgPool(1×1) → FC

Padding strategy:
  If H or W is not divisible by ws, we pad to the next multiple of ws before
  window/grid partitioning and crop back afterwards. This ensures correctness
  for arbitrary input sizes.
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.maxvit._config import MaxViTConfig

# ---------------------------------------------------------------------------
# Window partition / reverse
# ---------------------------------------------------------------------------


def _window_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """(B, H, W, C) → (B*nH*nW, ws, ws, C).

    Divides the spatial map into non-overlapping ws×ws windows.
    Requires H and W to be divisible by ws (caller must pad if needed).
    """
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    # (B, nH, ws, nW, ws, C) → permute → (B, nH, nW, ws, ws, C) → reshape
    x = x.reshape(B, nH, ws, nW, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, nH, nW


def _window_reverse(windows: Tensor, ws: int, nH: int, nW: int) -> Tensor:
    """(B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    B_total = windows.shape[0]
    B = B_total // (nH * nW)
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * ws, nW * ws, C)


# ---------------------------------------------------------------------------
# Grid partition / reverse
# ---------------------------------------------------------------------------


def _grid_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """(B, H, W, C) → (B*ws*ws, nH*nW, C).

    Grid attention picks every ws-th pixel to form a "grid group".
    Group (r, s) contains pixels at positions (i*ws+r, j*ws+s) for all i,j.
    We merge (r,s) into the batch dimension so each group attends globally.

    Requires H, W divisible by ws (caller must pad if needed).
    """
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    # (B, nH, ws, nW, ws, C) → permute (0,2,4,1,3,5) → (B, ws, ws, nH, nW, C)
    x = x.reshape(B, nH, ws, nW, ws, C)
    x = x.permute(0, 2, 4, 1, 3, 5)
    # Merge (B, ws, ws) → batch, flatten (nH, nW) → sequence
    x = x.reshape(B * ws * ws, nH * nW, C)
    return x, nH, nW


def _grid_reverse(x: Tensor, ws: int, nH: int, nW: int, B: int) -> Tensor:
    """(B*ws*ws, nH*nW, C) → (B, H, W, C)."""
    C = x.shape[-1]
    x = x.reshape(B, ws, ws, nH, nW, C)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, nH * ws, nW * ws, C)


# ---------------------------------------------------------------------------
# Helpers: pad spatial to multiple of ws, then crop back
# ---------------------------------------------------------------------------


def _pad_to_multiple(x_cl: Tensor, ws: int) -> tuple[Tensor, int, int, int, int]:
    """Pad (B, H, W, C) tensor so H and W are multiples of ws.

    Returns padded tensor and original (H, W) for later cropping.
    """
    B, H, W, C = x_cl.shape
    pH = (ws - H % ws) % ws
    pW = (ws - W % ws) % ws
    if pH > 0:
        pad_h = lucid.zeros(B, pH, W, C)
        x_cl = lucid.cat([x_cl, pad_h], dim=1)
    if pW > 0:
        Hp = x_cl.shape[1]
        pad_w = lucid.zeros(B, Hp, pW, C)
        x_cl = lucid.cat([x_cl, pad_w], dim=2)
    return x_cl, H, W, pH, pW


# ---------------------------------------------------------------------------
# MBConv block (mobile inverted bottleneck, pre-norm with BN)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """MBConv: BN pre-norm → expand Conv1×1 → GELU → DWConv3×3 → GELU → project.

    Residual connection around the whole block. Expand ratio = 4 (default).
    """

    def __init__(self, dim: int, expand_ratio: int = 4) -> None:
        super().__init__()
        mid = dim * expand_ratio
        self.norm = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, mid, 1)
        self.act1 = nn.GELU()
        self.dwconv = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(mid, dim, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = x
        x = cast(Tensor, self.norm(x))
        x = cast(Tensor, self.act1(cast(Tensor, self.conv1(x))))
        x = cast(Tensor, self.act2(cast(Tensor, self.dwconv(x))))
        x = cast(Tensor, self.conv2(x))
        return shortcut + x


# ---------------------------------------------------------------------------
# Shared attention + MLP block (pre-LN, used for window and grid attention)
# ---------------------------------------------------------------------------


class _AttnBlock(nn.Module):
    """Pre-LN MHA block operating on (B, N, C) token sequences."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        n = cast(Tensor, self.norm1(x))
        attn_out, _ = self.attn(n, n, n)
        x = x + attn_out
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# MaxViT block: MBConv → block-attn → grid-attn
# ---------------------------------------------------------------------------


class _MaxViTBlock(nn.Module):
    """Single MaxViT block = MBConv + window attention + grid attention.

    Operates on (B, C, H, W) tensors throughout (NCHW layout).
    Window and grid attention convert to channel-last internally for
    the partition operations, then convert back.
    """

    def __init__(
        self, dim: int, num_heads: int, window_size: int, mlp_ratio: float
    ) -> None:
        super().__init__()
        self.ws = window_size
        self.mbconv = _MBConv(dim)
        self.block_attn = _AttnBlock(dim, num_heads, mlp_ratio)
        self.grid_attn = _AttnBlock(dim, num_heads, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W)
        ws = self.ws

        # 1. MBConv local feature extraction
        x = cast(Tensor, self.mbconv(x))

        B, C, H, W = x.shape
        # Convert to channel-last for window/grid operations
        x_cl = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Pad to multiples of ws so partition is exact
        x_cl, orig_H, orig_W, _pH, _pW = _pad_to_multiple(x_cl, ws)
        Hp, Wp = x_cl.shape[1], x_cl.shape[2]

        # 2. Block (window) attention — local within ws×ws windows
        wins, nH, nW = _window_partition(x_cl, ws)  # (B*nH*nW, ws, ws, C)
        wins_seq = wins.reshape(-1, ws * ws, C)  # (B*nH*nW, ws², C)
        wins_seq = cast(Tensor, self.block_attn(wins_seq))
        wins = wins_seq.reshape(-1, ws, ws, C)
        x_cl = _window_reverse(wins, ws, nH, nW)  # (B, Hp, Wp, C)

        # 3. Grid attention — global among pixels at same grid offset
        grids, gH, gW = _grid_partition(x_cl, ws)  # (B*ws², nH*nW, C)
        grids = cast(Tensor, self.grid_attn(grids))
        x_cl = _grid_reverse(grids, ws, gH, gW, B)  # (B, Hp, Wp, C)

        # Crop back to original spatial size
        x_cl = x_cl[:, :orig_H, :orig_W, :]

        return x_cl.permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Downsampling between stages
# ---------------------------------------------------------------------------


class _MaxViTDownsample(nn.Module):
    """Strided 3×3 conv + BN downsampling between stages."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.bn(cast(Tensor, self.conv(x))))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_maxvit(cfg: MaxViTConfig) -> tuple[
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleList,
    list[FeatureInfo],
    int,
]:
    """Build shared MaxViT body: stem + stages + downsamplers.

    Stem: Conv3×3(s=2) → BN → GELU → Conv3×3(s=1) → 2× downsampled feature map.
    """
    stem_dim = 32
    stem = nn.Sequential(
        nn.Conv2d(cfg.in_channels, stem_dim, 3, stride=2, padding=1),
        nn.BatchNorm2d(stem_dim),
        nn.GELU(),
        nn.Conv2d(stem_dim, cfg.dims[0], 3, stride=1, padding=1),
    )

    stages: list[nn.Module] = []
    downsamplers: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 2  # stem applies 2× downsampling

    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
        blocks: list[nn.Module] = [
            _MaxViTBlock(dim, cfg.num_heads, cfg.window_size, cfg.mlp_ratio)
            for _ in range(depth)
        ]
        stages.append(nn.Sequential(*blocks))
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < len(cfg.depths) - 1:
            next_dim = cfg.dims[i + 1]
            downsamplers.append(_MaxViTDownsample(dim, next_dim))
            reduction *= 2

    return (
        stem,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        fi,
        cfg.dims[-1],
    )


# ---------------------------------------------------------------------------
# MaxViT backbone
# ---------------------------------------------------------------------------


class MaxViT(PretrainedModel, BackboneMixin):
    """MaxViT feature extractor — global avg-pooled final stage features."""

    config_class: ClassVar[type[MaxViTConfig]] = MaxViTConfig
    base_model_prefix: ClassVar[str] = "maxvit"

    def __init__(self, config: MaxViTConfig) -> None:
        super().__init__(config)
        stem, stages, downs, fi, out_dim = _build_maxvit(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
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
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# MaxViT for image classification
# ---------------------------------------------------------------------------


class MaxViTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """MaxViT with global avg-pool + FC classifier head."""

    config_class: ClassVar[type[MaxViTConfig]] = MaxViTConfig
    base_model_prefix: ClassVar[str] = "maxvit"

    def __init__(self, config: MaxViTConfig) -> None:
        super().__init__(config)
        stem, stages, downs, _, out_dim = _build_maxvit(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
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
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
