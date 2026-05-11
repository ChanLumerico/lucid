"""PVT v2 (Pyramid Vision Transformer v2) backbone and classifier.

Paper: "PVT v2: Improved Baselines with Pyramid Vision Transformer"
        Wang et al., 2022 — https://arxiv.org/abs/2106.13797

Key differences from PVT v1:
  1. Overlapping patch embedding (kernel=7, stride=4, padding=3 for stage 0;
     kernel=3, stride=2, padding=1 for subsequent stages) — no positional
     embeddings needed because the overlapping conv implicitly encodes position.
  2. MLP contains a depthwise Conv2d (3×3, same padding) between fc1 and fc2,
     providing spatial awareness within the MLP.
  3. Spatial Reduction Attention (SRA) unchanged from v1: K and V are reduced
     via a stride-sr_ratio Conv2d before MHA, keeping cost manageable at high
     resolutions.

Architecture (PVT v2-B1 / our 'pvt_tiny' default, 224 input):
  patch_embed: OverlapPatchEmbed(k=7,s=4) — top-level, used by stage 0
  Stage 0: blocks (heads=1, sr=8) + norm   — no downsample; uses top-level patch_embed
  Stage 1: downsample(k=3,s=2) + blocks (heads=2, sr=4) + norm
  Stage 2: downsample(k=3,s=2) + blocks (heads=5, sr=2) + norm
  Stage 3: downsample(k=3,s=2) + blocks (heads=8, sr=1) + norm
  Head   : global avg-pool → FC(512, num_classes)

timm key layout (pvt_v2_b1):
  patch_embed.proj.*  patch_embed.norm.*          — top-level
  stages.0.blocks.N.*  stages.0.norm.*            — stage 0 (no downsample)
  stages.1.downsample.proj/norm  stages.1.blocks.N.*  stages.1.norm.*
  ...
  head.weight  head.bias
"""

from typing import ClassVar, cast

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.pvt._config import PVTConfig

# ---------------------------------------------------------------------------
# Overlapping patch embedding (also used as "downsample" for stages 1-3)
# ---------------------------------------------------------------------------


class _OverlapPatchEmbed(nn.Module):
    """Overlapping patch embedding via strided Conv2d + LayerNorm.

    Returns spatial feature map (B, C, H', W') where H'=H/stride, W'=W/stride.
    The overlapping kernel (size > stride) means adjacent windows share context,
    implicitly encoding position so explicit positional embeddings are unnecessary.
    """

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        patch_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        padding = patch_size // 2
        self.proj = nn.Conv2d(
            in_ch, embed_dim, patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:  # type: ignore[override]
        x = cast(Tensor, self.proj(x))  # (B, C, H', W')
        B, C, H, W = x.shape
        # permute to (B, H'*W', C), apply LN, return
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = cast(Tensor, self.norm(x))
        return x, H, W


# ---------------------------------------------------------------------------
# MLP with depthwise conv (PVT v2 key change)
# ---------------------------------------------------------------------------


class _DWConvMLP(nn.Module):
    """Two-layer MLP with a DWConv after fc1 for spatial mixing.

    fc1 → reshape→ DWConv3×3 → reshape → GELU → fc2
    """

    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        B, _N, _C = x.shape
        x = cast(Tensor, self.fc1(x))
        # After fc1: (B, N, hidden) — reshape to spatial for DWConv
        hidden = x.shape[-1]
        x_2d = x.permute(0, 2, 1).reshape(B, hidden, H, W)
        x_2d = cast(Tensor, self.dwconv(x_2d))
        x = x_2d.reshape(B, hidden, H * W).permute(0, 2, 1)
        x = F.gelu(x)
        return cast(Tensor, self.fc2(x))


# ---------------------------------------------------------------------------
# Spatial Reduction Attention (SRA)
# ---------------------------------------------------------------------------


class _SRAttention(nn.Module):
    """MHA with optional spatial reduction on K and V.

    When sr_ratio > 1, K and V are computed from a spatially reduced feature
    map (via a stride-sr_ratio Conv2d), reducing the O(N²) cost for large N.
    """

    def __init__(self, dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None  # type: ignore[assignment]
            self.norm = None  # type: ignore[assignment]

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        B, N, C = x.shape
        head_dim = C // self.num_heads

        q = cast(Tensor, self.q(x))
        q = q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_2d = cast(Tensor, self.sr(x_2d))  # (B, C, H', W')
            x_2d = x_2d.flatten(2).permute(0, 2, 1)  # (B, H'*W', C)
            x_2d = cast(Tensor, self.norm(x_2d))
            kv_src = x_2d
        else:
            kv_src = x

        kv = cast(Tensor, self.kv(kv_src))
        N2 = kv_src.shape[1]
        kv = kv.reshape(B, N2, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, C)
        return cast(Tensor, self.proj(x))


# ---------------------------------------------------------------------------
# PVT v2 transformer block
# ---------------------------------------------------------------------------


class _PVTBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _SRAttention(dim, num_heads, sr_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _DWConvMLP(dim, mlp_ratio)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        x = x + cast(Tensor, self.attn(cast(Tensor, self.norm1(x)), H, W))  # type: ignore[arg-type]
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x)), H, W))  # type: ignore[arg-type]
        return x


# ---------------------------------------------------------------------------
# One PVT stage (blocks + norm only; downsample handled outside)
# ---------------------------------------------------------------------------


class _PVTStage(nn.Module):
    """One PVT v2 stage = optional downsample + N × Block + LayerNorm.

    For stage 0, there is no ``downsample`` sub-module — the patch embedding
    lives at top-level on the model (``self.patch_embed``).
    For stages 1-3, the patch embedding is stored as ``self.downsample`` to
    match the timm key layout (``stages.N.downsample.proj/norm``).
    """

    def __init__(
        self,
        in_ch: int,
        embed_dim: int,
        patch_size: int,
        stride: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        is_first: bool,
    ) -> None:
        super().__init__()
        # Stage 0: patch_embed lives on the model, not inside the stage.
        # Stages 1-3: patch_embed stored as downsample (timm key layout).
        if not is_first:
            self.downsample = _OverlapPatchEmbed(in_ch, embed_dim, patch_size, stride)
        self.blocks = nn.ModuleList(
            [_PVTBlock(embed_dim, num_heads, sr_ratio, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward_tokens(self, tokens: Tensor, H: int, W: int) -> tuple[Tensor, int, int]:
        """Run blocks + norm on pre-embedded tokens (B, N, C)."""
        for blk in self.blocks:
            tokens = cast(Tensor, blk(tokens, H, W))  # type: ignore[arg-type]
        tokens = cast(Tensor, self.norm(tokens))
        B, _, C = tokens.shape
        x_out = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out, H, W

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:  # type: ignore[override]
        # x is a spatial map (B, C_in, H_in, W_in).
        # For stage 0 this should NOT be called directly — call forward_tokens.
        # For stages 1-3, downsample first, then run blocks.
        tokens, H, W = cast(
            tuple[Tensor, int, int], self.downsample(x)  # type: ignore[attr-defined]
        )
        return self.forward_tokens(tokens, H, W)


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_pvt(cfg: PVTConfig) -> tuple[
    _OverlapPatchEmbed,  # top-level patch_embed (stage 0)
    nn.ModuleList,  # stages (0-indexed)
    list[FeatureInfo],
    int,  # final embed dim
]:
    """Build PVT v2 stages matching timm pvt_v2_bN key layout.

    timm layout:
      patch_embed  — top-level; handles stage-0 spatial downsampling
      stages.0     — blocks + norm only (no downsample submodule)
      stages.1-3   — downsample + blocks + norm
    """
    stages: list[nn.Module] = []
    fi: list[FeatureInfo] = []

    in_ch = cfg.in_channels
    reduction = 1

    # Build top-level patch_embed for stage 0
    patch_embed = _OverlapPatchEmbed(in_ch, cfg.embed_dims[0], patch_size=7, stride=4)
    reduction *= 4

    for i, (dim, depth, heads, sr, mlp_r) in enumerate(
        zip(cfg.embed_dims, cfg.depths, cfg.num_heads, cfg.sr_ratios, cfg.mlp_ratios)
    ):
        is_first = i == 0
        # patch_size/stride used by downsample in stages 1-3
        patch_size = 3
        stride = 2
        if not is_first:
            reduction *= stride
        stages.append(
            _PVTStage(
                in_ch=in_ch,
                embed_dim=dim,
                patch_size=patch_size,
                stride=stride,
                depth=depth,
                num_heads=heads,
                sr_ratio=sr,
                mlp_ratio=mlp_r,
                is_first=is_first,
            )
        )
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        in_ch = dim

    return patch_embed, nn.ModuleList(stages), fi, cfg.embed_dims[-1]


# ---------------------------------------------------------------------------
# PVT backbone
# ---------------------------------------------------------------------------


class PVT(PretrainedModel, BackboneMixin):
    """PVT v2 feature extractor — mean-pooled final-stage token features."""

    config_class: ClassVar[type[PVTConfig]] = PVTConfig
    base_model_prefix: ClassVar[str] = "pvt"

    def __init__(self, config: PVTConfig) -> None:
        super().__init__(config)
        self.patch_embed, self.stages, fi, out_dim = _build_pvt(config)
        self._feature_info = fi
        self._out_dim = out_dim

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        # Stage 0: use top-level patch_embed then stage blocks
        tokens, H, W = cast(tuple[Tensor, int, int], self.patch_embed(x))
        x_spatial, H, W = cast(
            tuple[Tensor, int, int],
            self.stages[0].forward_tokens(tokens, H, W),  # type: ignore[union-attr]
        )
        # Stages 1-3: each stage calls its own downsample internally
        for stage in list(self.stages)[1:]:
            x_spatial, H, W = cast(tuple[Tensor, int, int], stage(x_spatial))
        # x_spatial: (B, C, H, W) — global average pool to (B, C)
        return x_spatial.flatten(2).mean(dim=2)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# PVT for image classification
# ---------------------------------------------------------------------------


class PVTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """PVT v2 with global avg-pool + FC head."""

    config_class: ClassVar[type[PVTConfig]] = PVTConfig
    base_model_prefix: ClassVar[str] = "pvt"

    def __init__(self, config: PVTConfig) -> None:
        super().__init__(config)
        self.patch_embed, self.stages, _, out_dim = _build_pvt(config)
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        # Stage 0: use top-level patch_embed then stage blocks
        tokens, H, W = cast(tuple[Tensor, int, int], self.patch_embed(x))
        x_spatial, H, W = cast(
            tuple[Tensor, int, int],
            self.stages[0].forward_tokens(tokens, H, W),  # type: ignore[union-attr]
        )
        # Stages 1-3
        for stage in list(self.stages)[1:]:
            x_spatial, H, W = cast(tuple[Tensor, int, int], stage(x_spatial))
        # x_spatial: (B, C, H, W) — mean pool to (B, C)
        x_vec = x_spatial.flatten(2).mean(dim=2)
        logits = cast(Tensor, self.classifier(x_vec))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
