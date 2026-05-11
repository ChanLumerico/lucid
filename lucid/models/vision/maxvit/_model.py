"""MaxViT backbone and classifier (Tu et al., 2022).

Paper: "MaxViT: Multi-Axis Vision Transformer"

Key ideas:
  1. Multi-Axis attention: combine local window attention (block) and global
     grid attention in every block — O(n) overall complexity.
  2. MBConv (mobile inverted bottleneck) in each block for local features.
  3. Window (block) attention: partition spatial into non-overlapping windows,
     run MHA within each window.
  4. Grid attention: transpose partition (every window_size-th pixel in each
     direction forms a grid), run MHA within each grid.

Architecture (MaxViT-T, image=224, ws=8):
  Stem   : Conv3×3(s=2) → Conv3×3(32→64) → (112×112, 64)
  Stage 1: 2 × MaxViTBlock(64)  → Downsample → (56×56, 128)
  Stage 2: 2 × MaxViTBlock(128) → Downsample → (28×28, 256)
  Stage 3: 5 × MaxViTBlock(256) → Downsample → (14×14, 512)
  Stage 4: 2 × MaxViTBlock(512)
  Head   : AdaptiveAvgPool(1×1) → FC
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
# Window partition / reverse (same as Swin)
# ---------------------------------------------------------------------------


def _window_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """(B, H, W, C) → (B*nH*nW, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, H // ws, W // ws


def _window_reverse(windows: Tensor, ws: int, nH: int, nW: int) -> Tensor:
    """(B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    B_total = windows.shape[0]
    B = B_total // (nH * nW)
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * ws, nW * ws, C)


# ---------------------------------------------------------------------------
# Grid partition: pick every ws-th pixel to form "grid" tokens
# ---------------------------------------------------------------------------


def _grid_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """(B, H, W, C) → (B*ws*ws, nH*nW, C) grid partition.

    Grid attention: tokens at positions (i*ws+r, j*ws+s) for all i,j,
    fixed (r,s) form a grid. We group by (r,s) to get ws² groups of
    nH*nW tokens each.
    """
    B, H, W, C = x.shape
    nH = H // ws
    nW = W // ws
    # Reshape: (B, nH, ws, nW, ws, C) → permute so ws dims come first
    x = x.reshape(B, nH, ws, nW, ws, C)
    # (B, ws, ws, nH, nW, C) — each (r,s) pair is a grid
    x = x.permute(0, 2, 4, 1, 3, 5)
    # Merge batch with grid position: (B*ws*ws, nH*nW, C)
    x = x.reshape(B * ws * ws, nH * nW, C)
    return x, nH, nW


def _grid_reverse(x: Tensor, ws: int, nH: int, nW: int, B: int) -> Tensor:
    """(B*ws*ws, nH*nW, C) → (B, H, W, C)."""
    C = x.shape[-1]
    x = x.reshape(B, ws, ws, nH, nW, C)
    x = x.permute(0, 3, 1, 4, 2, 5)
    return x.reshape(B, nH * ws, nW * ws, C)


# ---------------------------------------------------------------------------
# MBConv block (mobile inverted bottleneck)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """MBConv: expand → DWConv3×3 → project, with pre-norm and residual."""

    def __init__(self, dim: int, expand_ratio: int = 4) -> None:
        super().__init__()
        mid = dim * expand_ratio
        self.norm = nn.BatchNorm2d(dim)
        self.conv1 = nn.Conv2d(dim, mid, 1)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(mid, mid, 3, padding=1, groups=mid)
        self.conv2 = nn.Conv2d(mid, dim, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = x
        x = cast(Tensor, self.norm(x))
        x = cast(Tensor, self.act(cast(Tensor, self.conv1(x))))
        x = cast(Tensor, self.act(cast(Tensor, self.dwconv(x))))
        x = cast(Tensor, self.conv2(x))
        return shortcut + x


# ---------------------------------------------------------------------------
# Attention block (shared for window and grid)
# ---------------------------------------------------------------------------


class _AttnBlock(nn.Module):
    """Pre-LN MHA block operating on (B, N, C) sequences."""

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
# MaxViT block: MBConv + block-attn + grid-attn
# ---------------------------------------------------------------------------


class _MaxViTBlock(nn.Module):
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

        # 1. MBConv (local feature extraction)
        x = cast(Tensor, self.mbconv(x))

        B, C, H, W = x.shape
        # Convert to channel-last for window/grid ops
        x_cl = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Pad H and W to be multiples of ws (required for partition ops).
        # At later stages (e.g. 28×28 with ws=8) the spatial dim may not
        # divide evenly; we pad with zeros and crop the output back.
        pH = (ws - H % ws) % ws
        pW = (ws - W % ws) % ws
        if pH > 0:
            pad_h = lucid.zeros(B, pH, W, C)
            x_cl = lucid.cat([x_cl, pad_h], dim=1)
        if pW > 0:
            Hp = x_cl.shape[1]
            pad_w = lucid.zeros(B, Hp, pW, C)
            x_cl = lucid.cat([x_cl, pad_w], dim=2)

        # 2. Block attention (local window)
        wins, nH, nW = _window_partition(x_cl, ws)  # (B*nH*nW, ws, ws, C)
        wins_flat = wins.reshape(-1, ws * ws, C)  # (B*nH*nW, ws², C)
        wins_flat = cast(Tensor, self.block_attn(wins_flat))
        wins = wins_flat.reshape(-1, ws, ws, C)
        x_cl = _window_reverse(wins, ws, nH, nW)  # (B, H_pad, W_pad, C)

        # 3. Grid attention (global sparse)
        grids, gH, gW = _grid_partition(x_cl, ws)  # (B*ws², nH*nW, C)
        grids = cast(Tensor, self.grid_attn(grids))
        x_cl = _grid_reverse(grids, ws, gH, gW, B)  # (B, H_pad, W_pad, C)

        # Crop back to original spatial size
        x_cl = x_cl[:, :H, :W, :]

        return x_cl.permute(0, 3, 1, 2)  # back to (B, C, H, W)


# ---------------------------------------------------------------------------
# Downsampling between stages
# ---------------------------------------------------------------------------


class _MaxViTDownsample(nn.Module):
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
    stem_dim = 32
    stem = nn.Sequential(
        nn.Conv2d(cfg.in_channels, stem_dim, 3, stride=2, padding=1),
        nn.GELU(),
        nn.Conv2d(stem_dim, cfg.dims[0], 3, stride=1, padding=1),
        nn.GELU(),
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
    """MaxViT feature extractor."""

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
    """MaxViT with global avg-pool + FC head."""

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
