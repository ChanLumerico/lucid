"""Swin Transformer backbone and classifier (Liu et al., 2021).

Paper: "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"

Key ideas vs plain ViT:
  1. Hierarchical feature maps (4 stages, 2× spatial downsampling between stages).
  2. Local window attention — each token only attends within its W×W window.
  3. Shifted window partition (alternating between two offset grids) enables
     cross-window information flow without global attention.
  4. Relative position bias added to attention logits.

Architecture (Swin-T, image=224, patch=4, window=7):
  PatchEmbed : Conv2d(4×4, stride=4) → (56×56, 96)
  Stage 1    : 2 × SwinBlock(window) → PatchMerge → (28×28, 192)
  Stage 2    : 2 × SwinBlock(window) → PatchMerge → (14×14, 384)
  Stage 3    : 6 × SwinBlock(window) → PatchMerge → (7×7,  768)
  Stage 4    : 2 × SwinBlock(window)              → (7×7,  768)
  Head       : LayerNorm → AdaptiveAvgPool(1×1) → FC

Each SwinBlock:
  LayerNorm → WindowAttention (w/ rel-pos bias) → residual
  LayerNorm → MLP(GELU) → residual
  Alternating blocks use cyclic shift + mask to implement shifted windows.
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
from lucid.models.vision.swin._config import SwinConfig


# ---------------------------------------------------------------------------
# Patch embedding (non-overlapping, stride = patch_size)
# ---------------------------------------------------------------------------

class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> Tensor:
        x = cast(Tensor, self.proj(x))          # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)              # (B, H, W, C)
        x = cast(Tensor, self.norm(x))
        return x


# ---------------------------------------------------------------------------
# Patch merging (spatial 2× downsampling + channel 2× expansion)
# ---------------------------------------------------------------------------

class _PatchMerge(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> Tensor:
        # x: (B, H, W, C)
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = lucid.cat([x0, x1, x2, x3], dim=-1)   # (B, H/2, W/2, 4C)
        x = cast(Tensor, self.norm(x))
        return cast(Tensor, self.proj(x))           # (B, H/2, W/2, 2C)


# ---------------------------------------------------------------------------
# Window partition / reverse helpers
# ---------------------------------------------------------------------------

def _window_partition(x: Tensor, ws: int) -> tuple[Tensor, int, int]:
    """Split (B, H, W, C) into (num_windows*B, ws, ws, C)."""
    B, H, W, C = x.shape
    x = x.reshape(B, H // ws, ws, W // ws, ws, C)
    # (B, nH, ws, nW, ws, C) → (B*nH*nW, ws, ws, C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C)
    return x, H // ws, W // ws


def _window_reverse(windows: Tensor, ws: int, nH: int, nW: int) -> Tensor:
    """Reverse of _window_partition: (B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    B_total = windows.shape[0]
    B = B_total // (nH * nW)
    C = windows.shape[-1]
    x = windows.reshape(B, nH, nW, ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, nH * ws, nW * ws, C)


# ---------------------------------------------------------------------------
# Window Multi-Head Self-Attention with relative position bias
# ---------------------------------------------------------------------------

class _WindowAttention(nn.Module):
    """Local window attention with learnable relative position bias."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv   = nn.Linear(dim, dim * 3, bias=True)
        self.proj  = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(p=attn_drop)

        # Relative position bias table: (2W-1)^2 × num_heads
        n = (2 * window_size - 1) ** 2
        self.rel_pos_bias = nn.Parameter(lucid.zeros(n, num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        # Pre-compute relative position index as int64 (no in-place ops)
        coords_1d = lucid.arange(window_size).to(lucid.int64)
        gy, gx = lucid.meshgrid(coords_1d, coords_1d, indexing="ij")  # (ws, ws)
        flat_y, flat_x = gy.flatten(), gx.flatten()                    # (ws^2,)
        rel_y = flat_y.unsqueeze(1) - flat_y.unsqueeze(0) + (window_size - 1)  # (ws^2, ws^2)
        rel_x = flat_x.unsqueeze(1) - flat_x.unsqueeze(0) + (window_size - 1)
        rel_idx = rel_y * (2 * window_size - 1) + rel_x               # (ws^2, ws^2)
        self.rel_pos_idx: Tensor
        object.__setattr__(self, "rel_pos_idx", rel_idx)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        B_, N, C = x.shape                                       # B_ = num_windows*B
        qkv = cast(Tensor, self.qkv(x))
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale

        # Relative position bias
        idx = self.rel_pos_idx.reshape(-1)
        bias = self.rel_pos_bias[idx].reshape(
            self.ws * self.ws, self.ws * self.ws, self.num_heads
        ).permute(2, 0, 1).unsqueeze(0)
        attn = attn + bias

        if mask is not None:
            # mask: (num_windows, N, N) with -100 for masked positions
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = cast(Tensor, self.attn_drop(attn))

        x = (attn @ v).permute(0, 2, 1, 3).reshape(B_, N, C)
        return cast(Tensor, self.proj(x))


# ---------------------------------------------------------------------------
# Swin Transformer block
# ---------------------------------------------------------------------------

class _SwinBlock(nn.Module):
    """One Swin Transformer block (regular or shifted window)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift: bool,
        mlp_ratio: float,
        dropout: float,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.ws = window_size
        self.shift = shift
        self.shift_size = window_size // 2 if shift else 0

        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _WindowAttention(dim, window_size, num_heads, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim    = int(dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(p=dropout),
        )

    def _attn_mask(self, H: int, W: int) -> Tensor | None:
        if self.shift_size == 0:
            return None
        ws = self.ws
        ss = self.shift_size
        img_mask = lucid.zeros(1, H, W, 1)
        slices_h = [slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]
        slices_w = [slice(0, -ws), slice(-ws, -ss), slice(-ss, None)]
        cnt = 0
        for sh in slices_h:
            for sw in slices_w:
                img_mask[0, sh, sw, 0] = cnt
                cnt += 1
        mask_windows, nH, nW = _window_partition(img_mask, ws)        # (nW, ws, ws, 1)
        mask_windows = mask_windows.reshape(-1, ws * ws)               # (nW, ws^2)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # (nW, ws^2, ws^2)
        # Replace non-zero with -100
        mask = lucid.where(mask != 0, lucid.full(mask.shape, -100.0), lucid.zeros(mask.shape))
        return mask

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> Tensor:
        B, H, W, C = x.shape
        shortcut = x
        x = cast(Tensor, self.norm1(x))

        if self.shift_size > 0:
            ss = self.shift_size
            x = lucid.roll(x, [-ss, -ss], dims=[1, 2])  # type: ignore[list-item]

        mask = self._attn_mask(H, W)
        windows, nH, nW = _window_partition(x, self.ws)
        windows = windows.reshape(-1, self.ws * self.ws, C)

        attn_out = cast(Tensor, self.attn(windows, mask=mask))
        attn_out = attn_out.reshape(-1, self.ws, self.ws, C)
        x = _window_reverse(attn_out, self.ws, nH, nW)

        if self.shift_size > 0:
            ss = self.shift_size
            x = lucid.roll(x, [ss, ss], dims=[1, 2])  # type: ignore[list-item]

        x = shortcut + x
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Swin stage (sequence of blocks + optional patch merge)
# ---------------------------------------------------------------------------

class _SwinStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        dropout: float,
        attn_drop: float,
        downsample: bool,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([
            _SwinBlock(
                dim, num_heads, window_size,
                shift=(i % 2 == 1),
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        self.downsample: nn.Module | None = _PatchMerge(dim) if downsample else None

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> Tensor:
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        if self.downsample is not None:
            x = cast(Tensor, self.downsample(x))
        return x


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------

def _build_swin(cfg: SwinConfig) -> tuple[
    _PatchEmbed, nn.ModuleList, nn.LayerNorm, list[FeatureInfo], int
]:
    patch_embed = _PatchEmbed(cfg.in_channels, cfg.patch_size, cfg.embed_dim)

    stages: list[nn.Module] = []
    dim = cfg.embed_dim
    fi: list[FeatureInfo] = []
    reduction = cfg.patch_size

    for i, (depth, heads) in enumerate(zip(cfg.depths, cfg.num_heads)):
        downsample = (i < len(cfg.depths) - 1)
        stages.append(_SwinStage(
            dim, depth, heads, cfg.window_size,
            cfg.mlp_ratio, cfg.dropout, cfg.attention_dropout,
            downsample,
        ))
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        if downsample:
            reduction *= 2
            dim *= 2

    norm = nn.LayerNorm(dim)
    return patch_embed, nn.ModuleList(stages), norm, fi, dim


# ---------------------------------------------------------------------------
# Swin Transformer backbone  (task="base")
# ---------------------------------------------------------------------------

class SwinTransformer(PretrainedModel, BackboneMixin):
    """Swin Transformer feature extractor — outputs (B, C) global avg-pooled feature."""

    config_class: ClassVar[type[SwinConfig]] = SwinConfig
    base_model_prefix: ClassVar[str] = "swin"

    def __init__(self, config: SwinConfig) -> None:
        super().__init__(config)
        pe, stages, norm, fi, out_dim = _build_swin(config)
        self.patch_embed = pe
        self.stages      = stages
        self.norm        = norm
        self._feature_info = fi
        self._out_dim    = out_dim
        self.avgpool     = nn.AdaptiveAvgPool2d(1)

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.patch_embed(x))   # (B, H/p, W/p, C)
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        x = cast(Tensor, self.norm(x))   # (B, H', W', C)
        # Global average pool: permute to (B, C, H', W') → avgpool → flatten
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)       # (B, C, H', W')
        x = cast(Tensor, self.avgpool(x)).flatten(1)  # (B, C)
        return x

    def forward(  # type: ignore[override]
        self, x: Tensor
    ) -> BaseModelOutput:
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# Swin Transformer for image classification  (task="image-classification")
# ---------------------------------------------------------------------------

class SwinTransformerForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Swin Transformer with global average pool + FC classification head."""

    config_class: ClassVar[type[SwinConfig]] = SwinConfig
    base_model_prefix: ClassVar[str] = "swin"

    def __init__(self, config: SwinConfig) -> None:
        super().__init__(config)
        pe, stages, norm, _, out_dim = _build_swin(config)
        self.patch_embed = pe
        self.stages      = stages
        self.norm        = norm
        self.avgpool     = nn.AdaptiveAvgPool2d(1)
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.patch_embed(x))
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        x = cast(Tensor, self.norm(x))
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)
        x = cast(Tensor, self.avgpool(x)).flatten(1)
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
