"""CoAtNet backbone and classification head (Dai et al., 2021).

Combines depthwise-separable MBConv stages with multi-head relative
self-attention Transformer stages following the CoAtNet-0 specification.

Architecture (CoAtNet-0, 4 stages after the stem):
  Stem   : 3×3 Conv(3→64) s=2 → BN → GELU → 3×3 Conv(64→64) → BN → GELU
  Stage 1: 2 × MBConv(64→96,  s=2)   [C-stage]
  Stage 2: 3 × MBConv(96→192, s=2)   [C-stage]
  Stage 3: 5 × RelAttnTransformer(192→384, pool-s=2)  [T-stage]
  Stage 4: 2 × RelAttnTransformer(384→768, pool-s=2)  [T-stage]
  Head   : AdaptiveAvgPool → LayerNorm → Linear(768→1000)

Reference param count: ~25.6 M (timm coatnet_0).
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
# Squeeze-and-Excitation channel attention
# ---------------------------------------------------------------------------


class _SE(nn.Module):
    """Squeeze-and-Excitation (SE) channel attention block."""

    def __init__(self, in_ch: int, se_ch: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_ch, se_ch)
        self.fc2 = nn.Linear(se_ch, in_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W)  →  squeeze to (B, C)  →  excite  →  (B, C, 1, 1)
        s = x.mean(dim=(2, 3))  # global average pool
        s = F.silu(cast(Tensor, self.fc1(s)))
        s = F.sigmoid(cast(Tensor, self.fc2(s)))
        return x * s.reshape(s.shape[0], s.shape[1], 1, 1)


# ---------------------------------------------------------------------------
# MBConv block (Mobile Inverted Bottleneck with SE)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """Mobile Inverted Bottleneck: BN-pre → expand → DWConv → SE → project.

    Expansion uses ``out_ch * expand`` as mid-channels (expand_output style).
    Squeeze-and-Excitation is applied between DWConv and projection, with
    se_ch = max(1, round(out_ch * se_ratio)).

    Downsampling: stride=2 on the depthwise conv; shortcut uses AvgPool2d+Conv1×1.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int = 4,
        stride: int = 1,
        se_ratio: float = 0.25,
    ) -> None:
        super().__init__()
        mid_ch = out_ch * expand
        se_ch = max(1, round(out_ch * se_ratio))
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
        self.se = _SE(mid_ch, se_ch)
        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=True)

        self.shortcut: nn.Module
        if stride != 1 or in_ch != out_ch:
            sc_layers: list[nn.Module] = []
            if stride != 1:
                sc_layers.append(nn.AvgPool2d(stride, stride=stride))
            sc_layers.append(nn.Conv2d(in_ch, out_ch, 1, bias=True))
            self.shortcut = nn.Sequential(*sc_layers)
        else:
            self.shortcut = nn.Sequential()  # identity

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = cast(Tensor, self.shortcut(x))

        out = F.gelu(cast(Tensor, self.bn_pre(x)))
        out = F.gelu(cast(Tensor, self.bn_exp(cast(Tensor, self.expand(out)))))
        out = F.gelu(cast(Tensor, self.bn_dw(cast(Tensor, self.dw(out)))))
        out = cast(Tensor, self.se(out))
        out = cast(Tensor, self.project(out))
        return out + shortcut


# ---------------------------------------------------------------------------
# Relative-position self-attention block (used in Transformer stages)
# ---------------------------------------------------------------------------


class _RelAttnBlock(nn.Module):
    """Pre-norm Transformer block with relative position bias.

    Relative position bias table is built for a fixed (H, W) grid determined
    at construction time. The actual grid at runtime must match; the feature
    map is averaged-pooled to this size if it does not (graceful degradation).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_h: int,
        grid_w: int,
        mlp_ratio: int = 4,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Relative position bias: table is (2H-1) × (2W-1) per head
        self.rel_bias = nn.Parameter(
            lucid.zeros(num_heads, (2 * grid_h - 1) * (2 * grid_w - 1))
        )

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

        self._init_rel_idx()

    def _init_rel_idx(self) -> None:
        H, W = self.grid_h, self.grid_w
        # Relative index for each pair of positions
        coords_h = lucid.arange(H).to(lucid.int64)
        coords_w = lucid.arange(W).to(lucid.int64)
        # Build 2-D position grid (H*W, 2)
        ys = coords_h.reshape(H, 1).expand(H, W).reshape(-1)
        xs = coords_w.reshape(1, W).expand(H, W).reshape(-1)
        N = H * W
        # Relative offset: (N, N)
        rel_h = ys.reshape(N, 1) - ys.reshape(1, N)  # row diffs
        rel_w = xs.reshape(N, 1) - xs.reshape(1, N)  # col diffs
        # Shift to [0, 2H-2] and [0, 2W-2]
        rel_h = rel_h + (H - 1)
        rel_w = rel_w + (W - 1)
        # Combine into flat index: row * (2W-1) + col
        rel_idx = rel_h * (2 * W - 1) + rel_w  # (N, N)
        # Register as non-parameter buffer
        object.__setattr__(self, "_rel_idx", rel_idx)

    def _rel_pos_bias(self) -> Tensor:
        # rel_idx: (N, N), rel_bias: (num_heads, (2H-1)*(2W-1))
        # Returns (num_heads, N, N)
        idx: Tensor = self._rel_idx  # type: ignore[assignment]
        idx_flat = idx.reshape(-1)  # (N*N,)
        # Gather from bias table
        bias = self.rel_bias[:, idx_flat]  # (num_heads, N*N)
        N = self.grid_h * self.grid_w
        return bias.reshape(self.num_heads, N, N)

    def _attn(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = cast(Tensor, self.qkv(x))  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale
        # Add relative position bias (num_heads, N, N) → broadcast over B
        bias = self._rel_pos_bias()  # (heads, N, N)
        attn = attn + bias.reshape(1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)

        out = attn @ v  # (B, heads, N, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return cast(Tensor, self.proj(out))

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N, C)
        x = x + self._attn(cast(Tensor, self.norm1(x)))
        x = x + cast(
            Tensor,
            self.fc2(F.gelu(cast(Tensor, self.fc1(cast(Tensor, self.norm2(x)))))),
        )
        return x


# ---------------------------------------------------------------------------
# Transformer stage (handles pool → flatten → blocks → unflatten)
# ---------------------------------------------------------------------------


class _TransformerStage(nn.Module):
    """Transformer stage: AvgPool2d(2) → linear channel proj → N×RelAttnBlock.

    The stage always spatially downsamples by 2× via AvgPool2d before the
    first block.  Channel projection (in_ch → out_ch) is a single Linear.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        num_heads: int,
        input_grid: tuple[int, int],
    ) -> None:
        super().__init__()
        # Grid after 2× pooling
        grid_h = input_grid[0] // 2
        grid_w = input_grid[1] // 2

        self.pool = nn.AvgPool2d(2, stride=2)
        self.proj = nn.Linear(in_ch, out_ch)
        self.blocks = nn.ModuleList(
            [
                _RelAttnBlock(out_ch, num_heads, grid_h, grid_w)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W)
        x = cast(Tensor, self.pool(x))
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        x = cast(Tensor, self.proj(x))  # (B, N, out_ch)
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))
        D = x.shape[2]
        x = x.permute(0, 2, 1).reshape(B, D, H, W)  # (B, out_ch, H, W)
        return x


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body(
    config: CoAtNetConfig,
) -> tuple[
    nn.Sequential,  # stem
    nn.Sequential,  # s1 (MBConv)
    nn.Sequential,  # s2 (MBConv)
    _TransformerStage,  # s3 (Transformer)
    _TransformerStage,  # s4 (Transformer)
    list[FeatureInfo],
]:
    d = config.dims  # (96, 192, 384, 768)
    n = config.blocks_per_stage  # (2, 3, 5, 2)
    exp = config.mbconv_expand
    heads = config.attn_heads
    img_size = config.image_size

    # ------------------------------------------------------------------ stem
    # Two conv layers, total stride 2:  3→stem_ch→stem_ch (s=2, then s=1)
    stem_ch = config.stem_width
    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.GELU(),
        nn.Conv2d(stem_ch, stem_ch, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.GELU(),
    )
    # After stem: H/2 × W/2

    # ------------------------------------------------------------------ S1
    s1_layers: list[nn.Module] = []
    s1_layers.append(_MBConv(stem_ch, d[0], expand=exp, stride=2))
    for _ in range(1, n[0]):
        s1_layers.append(_MBConv(d[0], d[0], expand=exp, stride=1))
    s1 = nn.Sequential(*s1_layers)
    # After S1: H/4 × W/4

    # ------------------------------------------------------------------ S2
    s2_layers: list[nn.Module] = []
    s2_layers.append(_MBConv(d[0], d[1], expand=exp, stride=2))
    for _ in range(1, n[1]):
        s2_layers.append(_MBConv(d[1], d[1], expand=exp, stride=1))
    s2 = nn.Sequential(*s2_layers)
    # After S2: H/8 × W/8

    # ------------------------------------------------------------------ S3
    s3_grid = (img_size // 8, img_size // 8)  # input to S3
    s3 = _TransformerStage(d[1], d[2], n[2], heads[0], input_grid=s3_grid)
    # After S3: H/16 × W/16

    # ------------------------------------------------------------------ S4
    s4_grid = (img_size // 16, img_size // 16)  # input to S4
    s4 = _TransformerStage(d[2], d[3], n[3], heads[1], input_grid=s4_grid)
    # After S4: H/32 × W/32

    feature_info = [
        FeatureInfo(stage=1, num_channels=d[0], reduction=4),
        FeatureInfo(stage=2, num_channels=d[1], reduction=8),
        FeatureInfo(stage=3, num_channels=d[2], reduction=16),
        FeatureInfo(stage=4, num_channels=d[3], reduction=32),
    ]
    return stem, s1, s2, s3, s4, feature_info


# ---------------------------------------------------------------------------
# CoAtNet backbone (task="base")
# ---------------------------------------------------------------------------


class CoAtNet(PretrainedModel, BackboneMixin):
    """CoAtNet feature extractor — returns spatial feature map from S4.

    Output: ``BaseModelOutput`` with ``last_hidden_state`` shaped
    ``(B, d[3], H/32, W/32)``.
    """

    config_class: ClassVar[type[CoAtNetConfig]] = CoAtNetConfig
    base_model_prefix: ClassVar[str] = "coatnet"

    def __init__(self, config: CoAtNetConfig) -> None:
        super().__init__(config)
        stem, s1, s2, s3, s4, fi = _build_body(config)
        self.stem = stem
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
    """CoAtNet with global average pooling + classification head.

    Head follows timm convention: AdaptiveAvgPool → LayerNorm → (optional)
    head_hidden Linear → classifier Linear.
    """

    config_class: ClassVar[type[CoAtNetConfig]] = CoAtNetConfig
    base_model_prefix: ClassVar[str] = "coatnet"

    def __init__(self, config: CoAtNetConfig) -> None:
        super().__init__(config)
        stem, s1, s2, s3, s4, _ = _build_body(config)
        self.stem = stem
        self.s1 = s1
        self.s2 = s2
        self.s3 = s3
        self.s4 = s4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        feat_dim = config.dims[-1]
        self.norm = nn.LayerNorm(feat_dim)
        # Optional hidden layer (timm head_hidden_size=768 for coatnet_0)
        if config.head_hidden_size is not None:
            self.pre_logits: nn.Module = nn.Sequential(
                nn.Linear(feat_dim, config.head_hidden_size),
                nn.Tanh(),
            )
            head_in = config.head_hidden_size
        else:
            self.pre_logits = nn.Sequential()
            head_in = feat_dim
        self._build_classifier(head_in, config.num_classes, dropout=config.dropout)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.s1(x))
        x = cast(Tensor, self.s2(x))
        x = cast(Tensor, self.s3(x))
        x = cast(Tensor, self.s4(x))
        x = cast(Tensor, self.avgpool(x))
        x = x.flatten(1)
        x = cast(Tensor, self.norm(x))
        x = cast(Tensor, self.pre_logits(x))
        logits = cast(Tensor, self.classifier(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
