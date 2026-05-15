"""EfficientFormer backbone and classifier (Li et al., 2022).

Paper: "EfficientFormer: Vision Transformers at MobileNet Speed"

Key ideas:
  1. MetaFormer insight: most of the capacity comes from the MLP, not the
     token mixer. Replace global attention with cheap local pooling in early
     stages.
  2. All-4D processing in early stages (B,C,H,W) — no reshape overhead.
  3. Stage 4 (last stage only): switch to standard MHA for global context.
  4. Depthwise conv stem (2×stride-2) for efficient spatial downsampling.

Architecture (EfficientFormer-L1, image=224):
  Stem   : Conv3×3(s=2) → Conv3×3(s=2) → (56×56, 48)
  Stage 1: 3 × PoolBlock(48)            → Downsample → (28×28, 96)
  Stage 2: 2 × PoolBlock(96)            → Downsample → (14×14, 224)
  Stage 3: 6 × PoolBlock(224)           → Downsample → (7×7,   448)
  Stage 4: 4 × AttnBlock(448)
  Head   : mean pool spatial → LN → FC
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models._utils._classification import DropPath, LayerScale
from lucid.models.vision.efficientformer._config import EfficientFormerConfig

# ---------------------------------------------------------------------------
# Pooling token mixer (MetaFormer-style)
# ---------------------------------------------------------------------------


class _PoolingBlock(nn.Module):
    """AvgPool3×3 − identity (pool the context, subtract self to get context diff)."""

    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.pool(x)) - x


# ---------------------------------------------------------------------------
# MLP (channel-last friendly)
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
# Stage 1-3 block: pooling-based (4D spatial tensors)
# ---------------------------------------------------------------------------


class _EfficientFormerPoolBlock(nn.Module):
    """Pooling MetaFormer block operating in (B, C, H, W) layout."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        drop_path_rate: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.pool_mixer = _PoolingBlock()
        self.norm = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # DWConv + BN (LePE-style local position encoding, no LayerScale).
        shortcut = x
        x = cast(Tensor, self.bn(cast(Tensor, self.dwconv(x))))
        x = shortcut + x

        # Pooling mixer (spatial) with LayerScale + DropPath.
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.ls1(self.pool_mixer(x)))))

        # MLP (channel-last) with LayerScale + DropPath. LayerScale applied
        # in (B, C, H, W) layout after permute-back.
        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1)
        x_cl = cast(Tensor, self.norm(x_cl))
        x_cl = cast(Tensor, self.mlp(x_cl))
        x_mlp = x_cl.permute(0, 3, 1, 2)
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.ls2(x_mlp))))
        return x


# ---------------------------------------------------------------------------
# Stage 4 block: attention-based (sequence layout)
# ---------------------------------------------------------------------------


class _EfficientFormerAttnBlock(nn.Module):
    """Standard MHA transformer block for stage 4."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop_path_rate: float,
        layer_scale_init: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio)
        self.ls1 = LayerScale(dim, layer_scale_init)
        self.ls2 = LayerScale(dim, layer_scale_init)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N, C)
        n = cast(Tensor, self.norm1(x))
        attn_out, _ = self.attn(n, n, n)
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.ls1(attn_out))))
        m = cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.ls2(m))))
        return x


# ---------------------------------------------------------------------------
# Downsampling between stages
# ---------------------------------------------------------------------------


class _Downsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.bn(cast(Tensor, self.conv(x))))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_efficientformer(cfg: EfficientFormerConfig) -> tuple[
    nn.Sequential,  # stem
    nn.ModuleList,  # stages (pool stages + attn stage)
    nn.ModuleList,  # downsamplers (len = num_stages - 1)
    nn.LayerNorm,  # head norm
    list[FeatureInfo],
    int,
]:
    # Stem: 2× Conv3×3 stride=2
    stem = nn.Sequential(
        nn.Conv2d(cfg.in_channels, cfg.embed_dims[0] // 2, 3, stride=2, padding=1),
        nn.BatchNorm2d(cfg.embed_dims[0] // 2),
        nn.ReLU(),
        nn.Conv2d(cfg.embed_dims[0] // 2, cfg.embed_dims[0], 3, stride=2, padding=1),
        nn.BatchNorm2d(cfg.embed_dims[0]),
        nn.ReLU(),
    )

    num_stages = len(cfg.depths)
    last_stage = num_stages - 1

    stages: list[nn.Module] = []
    downsamplers: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = 4  # stem applies 4× downsampling

    # Linear DropPath schedule across the whole trunk (paper §4.1).
    total_blocks = sum(cfg.depths)
    if total_blocks > 1 and cfg.drop_path_rate > 0.0:
        dp_rates = [
            cfg.drop_path_rate * i / (total_blocks - 1) for i in range(total_blocks)
        ]
    else:
        dp_rates = [cfg.drop_path_rate] * total_blocks
    cursor = 0

    for i, (depth, dim, mlp_ratio) in enumerate(
        zip(cfg.depths, cfg.embed_dims, cfg.mlp_ratios)
    ):
        if i < last_stage:
            # Pooling-based blocks (4D)
            stage = nn.Sequential(
                *[
                    _EfficientFormerPoolBlock(
                        dim, mlp_ratio, dp_rates[cursor + j], cfg.layer_scale_init
                    )
                    for j in range(depth)
                ]
            )
        else:
            # Attention-based blocks (sequence) — num_heads auto
            num_heads = max(1, dim // 32)
            stage = nn.Sequential(
                *[
                    _EfficientFormerAttnBlock(
                        dim,
                        num_heads,
                        mlp_ratio,
                        dp_rates[cursor + j],
                        cfg.layer_scale_init,
                    )
                    for j in range(depth)
                ]
            )
        cursor += depth
        stages.append(stage)
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))

        if i < num_stages - 1:
            downsamplers.append(_Downsample(dim, cfg.embed_dims[i + 1]))
            reduction *= 2

    head_norm = nn.LayerNorm(cfg.embed_dims[-1])
    return (
        stem,
        nn.ModuleList(stages),
        nn.ModuleList(downsamplers),
        head_norm,
        fi,
        cfg.embed_dims[-1],
    )


# ---------------------------------------------------------------------------
# EfficientFormer backbone
# ---------------------------------------------------------------------------


class EfficientFormer(PretrainedModel, BackboneMixin):
    """EfficientFormer feature extractor."""

    config_class: ClassVar[type[EfficientFormerConfig]] = EfficientFormerConfig
    base_model_prefix: ClassVar[str] = "efficientformer"

    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, fi, out_dim = _build_efficientformer(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._feature_info = fi
        self._out_dim = out_dim

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        num_stages = len(self.stages)
        last_stage = num_stages - 1

        for i, stage in enumerate(self.stages):
            if i < last_stage:
                # Pool stages: (B, C, H, W)
                x = cast(Tensor, stage(x))
            else:
                # Attention stage: flatten spatial → (B, N, C) → run → reshape back
                B, C, H, W = x.shape
                x_seq = x.flatten(2).permute(0, 2, 1)  # (B, N, C)
                x_seq = cast(Tensor, stage(x_seq))
                x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)

            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))

        # Global mean pool → (B, C)
        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # (B, N, C)
        x_cl = cast(Tensor, self.head_norm(x_cl))
        return x_cl.mean(dim=1)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# EfficientFormer for image classification
# ---------------------------------------------------------------------------


class EfficientFormerForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """EfficientFormer with mean-pool + LayerNorm + FC head."""

    config_class: ClassVar[type[EfficientFormerConfig]] = EfficientFormerConfig
    base_model_prefix: ClassVar[str] = "efficientformer"

    def __init__(self, config: EfficientFormerConfig) -> None:
        super().__init__(config)
        stem, stages, downs, hn, _, out_dim = _build_efficientformer(config)
        self.stem = stem
        self.stages = stages
        self.downsamplers = downs
        self.head_norm = hn
        self._build_classifier(out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        num_stages = len(self.stages)
        last_stage = num_stages - 1

        for i, stage in enumerate(self.stages):
            if i < last_stage:
                x = cast(Tensor, stage(x))
            else:
                B, C, H, W = x.shape
                x_seq = x.flatten(2).permute(0, 2, 1)
                x_seq = cast(Tensor, stage(x_seq))
                x = x_seq.permute(0, 2, 1).reshape(B, C, H, W)

            if i < len(self.downsamplers):
                x = cast(Tensor, self.downsamplers[i](x))

        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_cl = cast(Tensor, self.head_norm(x_cl))
        feat = x_cl.mean(dim=1)
        logits = cast(Tensor, self.classifier(feat))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
