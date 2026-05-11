"""PVT (Pyramid Vision Transformer) backbone and classifier (Wang et al., 2021).

Paper: "Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction
        without Convolutions"

Key ideas:
  1. Hierarchical 4-stage structure like ResNet, each stage downsamples spatially.
  2. Spatial Reduction Attention (SRA): K and V are pooled via Conv2d before MHA,
     reducing the quadratic cost of attention for high-resolution feature maps.
  3. Each stage: overlapping patch embedding → N transformer blocks → flatten.
  4. Position encodings reshaped per-stage based on current spatial resolution.

Architecture (PVT-Tiny, image=224):
  Stage 1: patch=4 → (56×56, 64),  2 × SRABlock(heads=1, sr=8)
  Stage 2: patch=2 → (28×28, 128), 2 × SRABlock(heads=2, sr=4)
  Stage 3: patch=2 → (14×14, 320), 2 × SRABlock(heads=5, sr=2)
  Stage 4: patch=2 → (7×7,  512),  2 × SRABlock(heads=8, sr=1)
  Head   : LayerNorm → mean over tokens → FC(512, num_classes)
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
from lucid.models.vision.pvt._config import PVTConfig

# ---------------------------------------------------------------------------
# Patch embedding (overlapping via stride = patch_size)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """Patch embedding: Conv2d(patch_size, stride=patch_size) → LN."""

    def __init__(
        self, in_ch: int, patch_size: int, embed_dim: int, img_size: int
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        self.h_patches = img_size // patch_size
        self.w_patches = img_size // patch_size

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:  # type: ignore[override]
        x = cast(Tensor, self.proj(x))  # (B, C, H, W)
        B, C, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, H*W, C)
        x = cast(Tensor, self.norm(x))
        return x, H, W


# ---------------------------------------------------------------------------
# Spatial Reduction Attention (SRA)
# ---------------------------------------------------------------------------


class _SRAttention(nn.Module):
    """MHA with optional spatial reduction on K and V."""

    def __init__(self, dim: int, num_heads: int, sr_ratio: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, sr_ratio, stride=sr_ratio)
        else:
            self.sr = None  # type: ignore[assignment]

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        B, N, C = x.shape
        head_dim = C // self.num_heads

        q = cast(Tensor, self.q(x))
        q = q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        if self.sr is not None:
            # Reshape tokens back to spatial form, apply stride conv to reduce
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
# MLP
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.fc2(F.gelu(cast(Tensor, self.fc1(x)))))


# ---------------------------------------------------------------------------
# PVT transformer block
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
        self.mlp = _MLP(dim, mlp_ratio)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        x = x + cast(Tensor, self.attn(cast(Tensor, self.norm1(x)), H, W))  # type: ignore[arg-type]
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# One PVT stage
# ---------------------------------------------------------------------------


class _PVTStage(nn.Module):
    def __init__(
        self,
        in_ch: int,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float,
        img_size: int,
    ) -> None:
        super().__init__()
        self.patch_embed = _PatchEmbed(in_ch, patch_size, embed_dim, img_size)
        self.pos_embed = nn.Parameter(
            lucid.zeros(
                1,
                (img_size // patch_size) * (img_size // patch_size),
                embed_dim,
            )
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [_PVTBlock(embed_dim, num_heads, sr_ratio, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:  # type: ignore[override]
        x, H, W = cast(tuple[Tensor, int, int], self.patch_embed(x))
        # Add positional embedding (may need to interpolate for different sizes)
        B, N, C = x.shape
        pos_enc: Tensor = self.pos_embed
        if pos_enc.shape[1] != N:
            # Interpolate positional encoding to match actual H, W
            pH = pW = int(math.isqrt(pos_enc.shape[1]))
            pos_2d = pos_enc.reshape(1, pH, pW, C).permute(0, 3, 1, 2)  # (1, C, pH, pW)
            pos_2d = F.interpolate(pos_2d, size=(H, W), mode="bilinear")
            pos_enc = pos_2d.permute(0, 2, 3, 1).reshape(1, H * W, C)
        x = x + pos_enc
        for blk in self.blocks:
            x = cast(Tensor, blk(x, H, W))  # type: ignore[arg-type]
        x = cast(Tensor, self.norm(x))
        # Reshape back to (B, C, H, W) for the next stage's patch embed
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x_2d, H, W


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_pvt(cfg: PVTConfig) -> tuple[nn.ModuleList, list[FeatureInfo], int]:
    stages: list[nn.Module] = []
    fi: list[FeatureInfo] = []

    # Stage spatial sizes for 224 input
    patch_sizes = [4, 2, 2, 2]
    # Cumulative reductions: 4, 8, 16, 32
    img_sizes = [224, 56, 28, 14]  # input size to each stage

    in_ch = cfg.in_channels
    reduction = 1

    for i, (dim, depth, heads, sr) in enumerate(
        zip(cfg.embed_dims, cfg.depths, cfg.num_heads, cfg.sr_ratios)
    ):
        patch_size = patch_sizes[i]
        img_size = img_sizes[i]
        reduction *= patch_size
        stages.append(
            _PVTStage(in_ch, patch_size, dim, depth, heads, sr, cfg.mlp_ratio, img_size)
        )
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        in_ch = dim

    return nn.ModuleList(stages), fi, cfg.embed_dims[-1]


# ---------------------------------------------------------------------------
# PVT backbone
# ---------------------------------------------------------------------------


class PVT(PretrainedModel, BackboneMixin):
    """PVT feature extractor — mean-pooled final-stage token features."""

    config_class: ClassVar[type[PVTConfig]] = PVTConfig
    base_model_prefix: ClassVar[str] = "pvt"

    def __init__(self, config: PVTConfig) -> None:
        super().__init__(config)
        stages, fi, out_dim = _build_pvt(config)
        self.stages = stages
        self._feature_info = fi
        self._out_dim = out_dim

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x, _, _ = cast(tuple[Tensor, int, int], stage(x))
        # x: (B, C, H, W) — mean pool to (B, C)
        return x.flatten(2).mean(dim=2)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# PVT for image classification
# ---------------------------------------------------------------------------


class PVTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """PVT with token mean-pooling + LayerNorm + FC head."""

    config_class: ClassVar[type[PVTConfig]] = PVTConfig
    base_model_prefix: ClassVar[str] = "pvt"

    def __init__(self, config: PVTConfig) -> None:
        super().__init__(config)
        stages, _, out_dim = _build_pvt(config)
        self.stages = stages
        self.norm = nn.LayerNorm(out_dim)
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        for stage in self.stages:
            x, _, _ = cast(tuple[Tensor, int, int], stage(x))
        # x: (B, C, H, W)
        x = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
        x = cast(Tensor, self.norm(x))
        x = x.mean(dim=1)  # (B, C) global average over tokens
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
