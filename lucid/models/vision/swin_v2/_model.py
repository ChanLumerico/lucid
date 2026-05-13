"""Swin Transformer V2 backbone and classifier (Liu et al., 2022).

Paper: "Swin Transformer V2: Scaling Up Capacity and Resolution"

Three key changes relative to V1:
  1. Scaled cosine self-attention — Q/K unit-normalised, temperature τ
     (learnable per-head log-scale) replaces the fixed 1/√d scale factor.
  2. Log-spaced Continuous Position Bias (CPB) — a small 2-layer MLP maps
     sign-log-spaced relative coordinates to per-head biases, replacing the
     learned embedding table.  Enables window-size transfer at test time.
  3. Post-normalization — LayerNorm applied *after* attention and MLP
     residual branches (instead of before, as in V1).

Architecture (SwinV2-T, image=256, patch=4, window=8):
  PatchEmbed : Conv2d(4×4, stride=4) → (64×64, 96)
  Stage 1    : 2  × SwinBlockV2 → PatchMerge → (32×32, 192)
  Stage 2    : 2  × SwinBlockV2 → PatchMerge → (16×16, 384)
  Stage 3    : 6  × SwinBlockV2 → PatchMerge → (8×8,  768)
  Stage 4    : 2  × SwinBlockV2              → (8×8,  768)
  Head       : LayerNorm → AdaptiveAvgPool(1×1) → FC
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
from lucid.models.vision.swin_v2._config import SwinV2Config

# ---------------------------------------------------------------------------
# Patch embedding — identical to V1 (non-overlapping, stride = patch_size)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, patch_size: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.proj(x))  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        return cast(Tensor, self.norm(x))


# ---------------------------------------------------------------------------
# Patch merging — identical to V1 (spatial 2× downsampling + channel ×2)
# ---------------------------------------------------------------------------


class _PatchMerge(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.proj = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = lucid.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = cast(Tensor, self.norm(x))
        return cast(Tensor, self.proj(x))  # (B, H/2, W/2, 2C)


# ---------------------------------------------------------------------------
# Window partition / reverse helpers — identical to V1
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
# MLP feed-forward block — identical structure to V1
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.net(x))


# ---------------------------------------------------------------------------
# V2 Change 2: Log-spaced Continuous Position Bias (CPB) helper
# ---------------------------------------------------------------------------


def _build_log_cpb_coords(window_size: int) -> list[list[float]]:
    """Return flat list of (lh, lw) log-space relative coords.

    For a window of size ws×ws every token pair (i,j) gets
      sign(d) * log(1 + |d|)   where d ∈ {-(ws-1), …, ws-1}.

    Output shape after lucid.tensor: (ws*ws * ws*ws, 2).
    """
    ws = window_size
    coords: list[list[float]] = []
    for hi in range(ws):
        for wi in range(ws):
            for hj in range(ws):
                for wj in range(ws):
                    dh = hi - hj
                    dw = wi - wj
                    lh = math.copysign(math.log1p(abs(dh)), dh) if dh != 0 else 0.0
                    lw = math.copysign(math.log1p(abs(dw)), dw) if dw != 0 else 0.0
                    coords.append([lh, lw])
    return coords


# ---------------------------------------------------------------------------
# V2 Change 1 + 2: Scaled Cosine Window Attention with CPB position bias
# ---------------------------------------------------------------------------


class _WindowAttentionV2(nn.Module):
    """Scaled cosine self-attention with log-spaced Continuous Position Bias.

    Replaces V1's dot-product attention + learned relative-position table.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads

        # V2 Change 1: learnable log-scale temperature τ per head.
        # Initialised at log(10); clamped to [0, log(100)] during forward.
        self.logit_scale = nn.Parameter(lucid.full((num_heads, 1, 1), math.log(10.0)))

        # V2 Change 2: CPB MLP — 2 → 512 → num_heads (no bias on output layer)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, num_heads, bias=False),
        )

        # Precompute log-space relative coords and cache as a plain attribute.
        # Shape: (ws*ws*ws*ws, 2) — frozen list converted to tensor on first use.
        self._cpb_coords_list: list[list[float]] = _build_log_cpb_coords(window_size)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def _get_cpb_bias(self) -> Tensor:
        """Compute CPB position bias tensor.

        Returns shape (num_heads, N, N) where N = window_size².
        """
        ws = self.window_size
        N = ws * ws
        # (N*N, 2) — converted each forward call; coord values are precomputed
        coord_t = lucid.tensor(self._cpb_coords_list)
        # (N*N, num_heads) — run through the MLP
        bias = cast(Tensor, self.cpb_mlp(coord_t))
        # Scale to (-16, +16) range as described in the paper
        bias = F.sigmoid(bias) * 16.0
        # Reshape to (num_heads, N, N)
        return bias.reshape(N, N, self.num_heads).permute(2, 0, 1)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        B_, N, C = x.shape  # B_ = num_windows*B, N = ws*ws
        H = self.num_heads
        D = C // H

        qkv = cast(Tensor, self.qkv(x))
        qkv = qkv.reshape(B_, N, 3, H, D).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B_, H, N, D)

        # V2 Change 1: scaled cosine attention
        # Normalise Q and K to unit vectors along the head-dim axis
        q_norm = (q * q).sum(dim=-1, keepdim=True) ** 0.5
        k_norm = (k * k).sum(dim=-1, keepdim=True) ** 0.5
        q_unit = q / (q_norm + 1e-6)
        k_unit = k / (k_norm + 1e-6)

        # Temperature τ = exp(logit_scale), clamped so τ ≤ log(100)
        # logit_scale: (H, 1, 1), broadcast over (B_, H, N, N)
        scale = self.logit_scale.exp().clamp(0.0, math.log(100.0))
        attn = (q_unit @ k_unit.permute(0, 1, 3, 2)) * scale

        # V2 Change 2: add CPB position bias (H, N, N) → broadcast over B_
        cpb = self._get_cpb_bias()  # (H, N, N)
        attn = attn + cpb.unsqueeze(0)  # (1, H, N, N)

        # Shifted-window mask (identical logic to V1)
        if mask is not None:
            # mask: (num_windows, N, N) with -100 for masked pairs
            nW = mask.shape[0]
            attn = attn.reshape(B_ // nW, nW, H, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape(-1, H, N, N)

        attn = cast(Tensor, self.softmax(attn))
        attn = cast(Tensor, self.attn_drop(attn))

        x = attn @ v  # (B_, H, N, D)
        x = x.permute(0, 2, 1, 3).reshape(B_, N, C)
        return cast(Tensor, self.proj(x))


# ---------------------------------------------------------------------------
# V2 Change 3: Post-norm Swin block
# ---------------------------------------------------------------------------


class _SwinBlockV2(nn.Module):
    """Swin Transformer V2 block with post-normalization.

    V1 applied LayerNorm *before* each sub-layer (pre-norm).
    V2 applies LayerNorm *after* each sub-layer and its residual (post-norm).
    """

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

        self.attn = _WindowAttentionV2(dim, window_size, num_heads, attn_drop)
        self.norm1 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), dropout)
        self.norm2 = nn.LayerNorm(dim)

    def _attn_mask(self, H: int, W: int) -> Tensor | None:
        """Build shifted-window attention mask — same logic as V1."""
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
        mask_windows, nH, nW = _window_partition(img_mask, ws)
        mask_windows = mask_windows.reshape(-1, ws * ws)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = lucid.where(
            mask != 0,
            lucid.full(mask.shape, -100.0),
            lucid.zeros(mask.shape),
        )
        return mask

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        B, H, W, C = x.shape

        # ── cyclic shift ──────────────────────────────────────────────────
        if self.shift_size > 0:
            ss = self.shift_size
            x_shift = lucid.roll(x, [-ss, -ss], dims=[1, 2])  # type: ignore[list-item]
        else:
            x_shift = x

        # ── window partition & attention ──────────────────────────────────
        mask = self._attn_mask(H, W)
        windows, nH, nW = _window_partition(x_shift, self.ws)
        windows = windows.reshape(-1, self.ws * self.ws, C)

        attn_out = cast(Tensor, self.attn(windows, mask=mask))
        attn_out = attn_out.reshape(-1, self.ws, self.ws, C)
        x_shift = _window_reverse(attn_out, self.ws, nH, nW)

        # ── reverse cyclic shift ──────────────────────────────────────────
        if self.shift_size > 0:
            ss = self.shift_size
            x_shift = lucid.roll(x_shift, [ss, ss], dims=[1, 2])  # type: ignore[list-item]

        # V2 post-norm: x = x + LN(attn(x))
        x = x + cast(Tensor, self.norm1(x_shift))

        # ── MLP with post-norm ────────────────────────────────────────────
        x = x + cast(Tensor, self.norm2(cast(Tensor, self.mlp(x))))
        return x


# ---------------------------------------------------------------------------
# Swin V2 stage — sequence of blocks + optional patch merge
# ---------------------------------------------------------------------------


class _SwinStageV2(nn.Module):
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
        self.blocks = nn.ModuleList(
            [
                _SwinBlockV2(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=(i % 2 == 1),
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attn_drop=attn_drop,
                )
                for i in range(depth)
            ]
        )
        self.downsample: nn.Module | None = _PatchMerge(dim) if downsample else None

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        if self.downsample is not None:
            x = cast(Tensor, self.downsample(x))
        return x


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_swin_v2(
    cfg: SwinV2Config,
) -> tuple[_PatchEmbed, nn.ModuleList, nn.LayerNorm, list[FeatureInfo], int]:
    patch_embed = _PatchEmbed(cfg.in_channels, cfg.patch_size, cfg.embed_dim)

    stages: list[nn.Module] = []
    dim = cfg.embed_dim
    fi: list[FeatureInfo] = []
    reduction = cfg.patch_size

    for i, (depth, heads) in enumerate(zip(cfg.depths, cfg.num_heads)):
        downsample = i < len(cfg.depths) - 1
        stages.append(
            _SwinStageV2(
                dim=dim,
                depth=depth,
                num_heads=heads,
                window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio,
                dropout=cfg.dropout,
                attn_drop=cfg.attention_dropout,
                downsample=downsample,
            )
        )
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        if downsample:
            reduction *= 2
            dim *= 2

    norm = nn.LayerNorm(dim)
    return patch_embed, nn.ModuleList(stages), norm, fi, dim


# ---------------------------------------------------------------------------
# Swin Transformer V2 backbone  (task="base")
# ---------------------------------------------------------------------------


class SwinTransformerV2(PretrainedModel, BackboneMixin):
    """Swin Transformer V2 feature extractor — outputs (B, C) global avg-pooled feature."""

    config_class: ClassVar[type[SwinV2Config]] = SwinV2Config
    base_model_prefix: ClassVar[str] = "swin_v2"

    def __init__(self, config: SwinV2Config) -> None:
        super().__init__(config)
        pe, stages, norm, fi, out_dim = _build_swin_v2(config)
        self.patch_embed = pe
        self.stages = stages
        self.norm = norm
        self._feature_info = fi
        self._out_dim = out_dim
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.patch_embed(x))  # (B, H/p, W/p, C)
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        x = cast(Tensor, self.norm(x))  # (B, H', W', C)
        B, H, W, C = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, C, H', W')
        x = cast(Tensor, self.avgpool(x)).flatten(1)  # (B, C)
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# Swin Transformer V2 for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class SwinTransformerV2ForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """Swin Transformer V2 with global average pool + FC classification head."""

    config_class: ClassVar[type[SwinV2Config]] = SwinV2Config
    base_model_prefix: ClassVar[str] = "swin_v2"

    def __init__(self, config: SwinV2Config) -> None:
        super().__init__(config)
        pe, stages, norm, _, out_dim = _build_swin_v2(config)
        self.patch_embed = pe
        self.stages = stages
        self.norm = norm
        self.avgpool = nn.AdaptiveAvgPool2d(1)
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
