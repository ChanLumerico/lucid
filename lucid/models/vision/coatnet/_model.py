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

from typing import ClassVar, cast, final, override

import lucid
import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._utils._classification import DropPath
from lucid.models._base import PretrainedModel
from lucid.models._tasks import ImageClassificationModel
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

    @override
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

    Downsampling: stride=2 on the expansion conv (Eqn 5); shortcut pools then projects.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int = 4,
        stride: int = 1,
        se_ratio: float = 0.25,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        # Table 10 lists stochastic depth for every CoAtNet variant; the
        # reference wires it through both block types.
        self.drop_path = DropPath(drop_path_rate)
        mid_ch = out_ch * expand
        se_ch = max(1, round(out_ch * se_ratio))
        self.stride = stride

        self.bn_pre = nn.BatchNorm2d(in_ch)
        # Eqn (5) puts the stride on the expansion conv:
        # ``Conv(DepthConv(Conv(Norm(x), stride=2)))``.  Striding the
        # depthwise instead is the paper's "Strided DConv" *alternative*
        # (Table 9), not the configuration its reported numbers use.
        self.expand = nn.Conv2d(in_ch, mid_ch, 1, stride=stride, bias=False)
        self.bn_exp = nn.BatchNorm2d(mid_ch)
        self.dw = nn.Conv2d(
            mid_ch,
            mid_ch,
            3,
            stride=1,
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

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        shortcut = cast(Tensor, self.shortcut(x))

        # Pre-activation form is ``x + Module(Norm(x))`` — Norm only; the
        # reference's pre_norm is built with apply_act=False.
        out = cast(Tensor, self.bn_pre(x))
        out = F.gelu(cast(Tensor, self.bn_exp(cast(Tensor, self.expand(out)))))
        out = F.gelu(cast(Tensor, self.bn_dw(cast(Tensor, self.dw(out)))))
        out = cast(Tensor, self.se(out))
        out = cast(Tensor, self.project(out))
        return cast(Tensor, self.drop_path(out)) + shortcut


# ---------------------------------------------------------------------------
# Relative-position self-attention block (used in Transformer stages)
# ---------------------------------------------------------------------------


# The reference runs CoAtNet's transformer LayerNorms at 1e-6, not the
# framework default of 1e-5.  The gap only matters where the normalised
# variance is small, which is exactly where a deep transformer stack
# spends its time.
_LN_EPS: float = 1e-6


@final
class _RelAttnBlock(nn.Module):
    """Pre-norm Transformer block with relative position bias.

    Relative position bias table is built for a fixed (H, W) grid determined
    at construction time. The actual grid at runtime must match; the feature
    map is averaged-pooled to this size if it does not (graceful degradation).
    """

    _rel_idx: Tensor

    def __init__(
        self,
        dim: int,
        num_heads: int,
        grid_h: int,
        grid_w: int,
        mlp_ratio: int = 4,
        drop_path_rate: float = 0.0,
        out_dim: int | None = None,
        downsample: bool = False,
    ) -> None:
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)
        self.dim = dim
        self.out_dim = out_dim
        self.downsample = downsample
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.scale = self.head_dim**-0.5
        # The bias table is indexed on the *post*-pool grid, which is what
        # attention runs on.
        self.grid_h = grid_h
        self.grid_w = grid_w
        # Pre-pool grid, needed to fold the sequence back to 2-D for pooling.
        self._downsample = downsample
        self.in_h = grid_h * 2 if downsample else grid_h
        self.in_w = grid_w * 2 if downsample else grid_w

        # Relative position bias: table is (2H-1) × (2W-1) per head
        self.rel_bias = nn.Parameter(
            lucid.zeros(num_heads, (2 * grid_h - 1) * (2 * grid_w - 1))
        )

        # Eqn (4): ``x <- Proj(Pool(x)) + Attention(Pool(Norm(x)))``.  The
        # identity branch projects the pooled input to the new width, and the
        # residual branch normalises at the *input* width before pooling — so
        # the widening happens inside attention, not in a stage-level
        # projection applied before any block.
        self.norm1 = nn.LayerNorm(dim, eps=_LN_EPS)
        self.qkv = nn.Linear(dim, out_dim * 3)
        self.proj = nn.Linear(out_dim, out_dim)
        self.shortcut: nn.Module = (
            nn.Linear(dim, out_dim) if dim != out_dim else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(out_dim, eps=_LN_EPS)
        mlp_dim = out_dim * mlp_ratio
        self.fc1 = nn.Linear(out_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)

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
        # Register as a proper non-persistent buffer so ``.to(device=...)``
        # moves it alongside parameters and Metal forward stays on-device.
        self.register_buffer("_rel_idx", rel_idx, persistent=False)

    def resample_rel_pos(self, grid_h: int, grid_w: int) -> None:
        """Resize the relative-position table to a new attention grid.

        Section A.1: "When finetuned on a larger resolution, we simply use
        bi-linear interpolation to increase the size
        :math:`(2H-1) \times (2W-1)` to the desired size
        :math:`(2H'-1) \times (2W'-1)`."  Every headline CoAtNet result
        depends on this — CoAtNet-3/-4 pre-train at 224 and fine-tune at
        384/512 — so without it a checkpoint is welded to the resolution it
        was built at.  Rebuilding the model at the new ``image_size`` gives
        correctly *shaped* tables but throws the learned ones away.

        Args:
            grid_h: New attention-grid height.
            grid_w: New attention-grid width.
        """
        if (grid_h, grid_w) == (self.grid_h, self.grid_w):
            return
        old_h = 2 * self.grid_h - 1
        old_w = 2 * self.grid_w - 1
        table = self.rel_bias.reshape(1, self.num_heads, old_h, old_w)
        resized = F.interpolate(
            table,
            size=(2 * grid_h - 1, 2 * grid_w - 1),
            mode="bilinear",
            align_corners=False,
        )
        self.rel_bias = nn.Parameter(
            resized.reshape(self.num_heads, (2 * grid_h - 1) * (2 * grid_w - 1))
        )
        self.grid_h, self.grid_w = grid_h, grid_w
        self.in_h = grid_h * 2 if self._downsample else grid_h
        self.in_w = grid_w * 2 if self._downsample else grid_w
        self._init_rel_idx()

    def _rel_pos_bias(self) -> Tensor:
        # rel_idx: (N, N), rel_bias: (num_heads, (2H-1)*(2W-1))
        # Returns (num_heads, N, N)
        idx: Tensor = self._rel_idx
        idx_flat = idx.reshape(-1)  # (N*N,)
        # Gather from bias table
        bias = self.rel_bias[:, idx_flat]  # (num_heads, N*N)
        N = self.grid_h * self.grid_w
        return bias.reshape(self.num_heads, N, N)

    def _attn(self, x: Tensor) -> Tensor:
        B, N, _C = x.shape
        qkv = cast(Tensor, self.qkv(x))  # (B, N, 3 * out_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        # Fused SDPA with the relative-position bias as an additive mask —
        # softmax((q·kᵀ)·scale + bias)·v, never forming the (B,H,N,N) scores.
        bias = self._rel_pos_bias().reshape(1, self.num_heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(B, N, self.out_dim)
        return cast(Tensor, self.proj(out))

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N, C)
        identity = self._pool(x)
        shortcut = cast(Tensor, self.shortcut(identity))
        h = self._pool(cast(Tensor, self.norm1(x)))
        x = shortcut + cast(Tensor, self.drop_path1(self._attn(h)))
        x = x + cast(
            Tensor,
            self.drop_path2(
                cast(
                    Tensor,
                    self.fc2(
                        F.gelu(cast(Tensor, self.fc1(cast(Tensor, self.norm2(x)))))
                    ),
                )
            ),
        )
        return x

    def _pool(self, seq: Tensor) -> Tensor:
        """Max-pool a ``(B, N, C)`` sequence on its 2-D grid, per Eqn (4)."""
        if not self.downsample:
            return seq
        b = int(seq.shape[0])
        c = int(seq.shape[2])
        grid = seq.permute(0, 2, 1).reshape(b, c, self.in_h, self.in_w)
        grid = F.max_pool2d(grid, 2, stride=2)
        return grid.reshape(b, c, self.grid_h * self.grid_w).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Transformer stage (handles pool → flatten → blocks → unflatten)
# ---------------------------------------------------------------------------


@final
class _TransformerStage(nn.Module):
    """Transformer stage: optional MaxPool2d(2) → linear channel proj → N×RelAttnBlock.

    When ``downsample=True`` (default — what S3 and S4 use in CoAtNet-0..5)
    the stage spatially halves via AvgPool2d before the first block.  When
    ``downsample=False`` it preserves spatial dims and just channel-projects
    + applies the N attention blocks — the mode CoAtNet-6 / CoAtNet-7 use
    for the transformer part of the mixed S3 (spatial halving already
    happened inside the preceding MBConv sub-stage).  Channel projection
    (``in_ch → out_ch``) is a single Linear.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        num_blocks: int,
        num_heads: int,
        input_grid: tuple[int, int],
        downsample: bool = True,
        dpr: list[float] | None = None,
    ) -> None:
        super().__init__()
        # Grid after optional 2× pooling.
        if downsample:
            grid_h = input_grid[0] // 2
            grid_w = input_grid[1] // 2
        else:
            grid_h = input_grid[0]
            grid_w = input_grid[1]

        self.grid_h, self.grid_w = grid_h, grid_w
        self.pool: nn.Module = (
            # Eqn (4): "the standard max pooling of stride 2 is directly
            # applied to the input states of both branches".
            nn.MaxPool2d(2, stride=2)
            if downsample
            else nn.Identity()
        )
        rates = dpr if dpr is not None else [0.0] * num_blocks
        # Eqn (4) makes the down-sample part of the *first block's* residual
        # structure, not a stage-level step: pooling once up front and then
        # running plain blocks gives a different function, because the
        # identity branch never sees the projection paired with its own pool.
        self.blocks = nn.ModuleList(
            [
                _RelAttnBlock(
                    in_ch if i == 0 else out_ch,
                    num_heads,
                    grid_h,
                    grid_w,
                    drop_path_rate=rates[i],
                    out_dim=out_ch,
                    downsample=(downsample and i == 0),
                )
                for i in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(out_ch)

    @override
    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W) — flattened once; the first block pools if needed.
        B, C, H, W = x.shape
        seq = x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)
        for blk in self.blocks:
            seq = cast(Tensor, blk(seq))
        seq = cast(Tensor, self.norm(seq))
        D = int(seq.shape[2])
        out_h, out_w = self.grid_h, self.grid_w
        return seq.permute(0, 2, 1).reshape(B, D, out_h, out_w)


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_body(
    config: CoAtNetConfig,
) -> tuple[
    nn.Sequential,  # stem
    nn.Sequential,  # s1 (MBConv)
    nn.Sequential,  # s2 (MBConv)
    nn.Module,  # s3 (Transformer for variants 0..5; Sequential for mixed-S3 variants 6/7)
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
    # No norm/activation after the second conv: the first MBConv's pre-norm
    # supplies it immediately, and §A.1 is explicit that the architecture is
    # pre-activation throughout ("x <- x + Module(Norm(x))").  Normalising
    # here as well ran BN -> GELU -> BN -> GELU back to back, which is not a
    # no-op — the second BN renormalises an already-rectified distribution —
    # and left a dead stem_ch-sized affine pair.  timm's Stem is
    # conv1 -> norm_act -> conv2 for the same reason.
    stem = nn.Sequential(
        nn.Conv2d(config.in_channels, stem_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(stem_ch),
        nn.GELU(),
        nn.Conv2d(stem_ch, stem_ch, 3, stride=1, padding=1, bias=False),
    )
    # Stochastic depth ramps linearly with the global block index (Table 10
    # lists a per-variant rate); a dispenser keeps the cursor correct across
    # the branching S3 layouts below.
    _total_blocks = sum(n) + (
        config.mixed_s3[0] + config.mixed_s3[1] if config.mixed_s3 is not None else 0
    )
    _dpr = [
        config.drop_path_rate * k / max(_total_blocks - 1, 1)
        for k in range(_total_blocks)
    ]
    _cursor = [0]

    def _next_dp() -> float:
        rate = _dpr[_cursor[0]] if _cursor[0] < len(_dpr) else 0.0
        _cursor[0] += 1
        return rate

    # After stem: H/2 × W/2

    # ------------------------------------------------------------------ S1
    s1_layers: list[nn.Module] = []
    s1_layers.append(
        _MBConv(stem_ch, d[0], expand=exp, stride=2, drop_path_rate=_next_dp())
    )
    for _ in range(1, n[0]):
        s1_layers.append(
            _MBConv(d[0], d[0], expand=exp, stride=1, drop_path_rate=_next_dp())
        )
    s1 = nn.Sequential(*s1_layers)
    # After S1: H/4 × W/4

    # ------------------------------------------------------------------ S2
    s2_layers: list[nn.Module] = []
    s2_layers.append(
        _MBConv(d[0], d[1], expand=exp, stride=2, drop_path_rate=_next_dp())
    )
    for _ in range(1, n[1]):
        s2_layers.append(
            _MBConv(d[1], d[1], expand=exp, stride=1, drop_path_rate=_next_dp())
        )
    s2 = nn.Sequential(*s2_layers)
    # After S2: H/8 × W/8

    # ------------------------------------------------------------------ S3
    # Two code paths:
    #   * Uniform (variants 0..5): single transformer stage as before.
    #   * Mixed (variants 6/7): MBConv sub-stage (does the spatial halving)
    #     → 1×1 channel-expand → transformer sub-stage at the wider width.
    s3_grid = (img_size // 8, img_size // 8)  # input to S3 (H/8 × W/8)
    s3: nn.Module
    s3_out_ch: int
    if config.mixed_s3 is None:
        s3 = _TransformerStage(
            d[1],
            d[2],
            n[2],
            heads[0],
            input_grid=s3_grid,
            dpr=[_next_dp() for _ in range(n[2])],
        )
        s3_out_ch = d[2]
    else:
        L_mb, L_attn, D_attn = config.mixed_s3
        # MBConv sub-stage: d[1] → d[2] with stride-2 in the first block,
        # then L_mb − 1 more isotropic blocks at d[2].  Mirrors S1/S2 layout.
        s3_mb_layers: list[nn.Module] = [
            _MBConv(d[1], d[2], expand=exp, stride=2, drop_path_rate=_next_dp())
        ]
        for _ in range(1, L_mb):
            s3_mb_layers.append(
                _MBConv(d[2], d[2], expand=exp, stride=1, drop_path_rate=_next_dp())
            )
        # 1×1 channel-expand transition: d[2] → D_attn (no spatial change).
        # Paper §A.2 only says "double the hidden dimension"; we realise it
        # as the standard 1×1 conv + BN + GELU used elsewhere in CoAtNet.
        s3_expand = nn.Sequential(
            nn.Conv2d(d[2], D_attn, 1, bias=False),
            nn.BatchNorm2d(D_attn),
            nn.GELU(),
        )
        # Transformer sub-stage at width D_attn, grid 1/16 (no further pool).
        # Head count comes from config.attn_heads[0] — paper convention is
        # head_dim=32 so this stays D_attn/32 for paper-faithful configs but
        # the user can override on the config without forking the model.
        s3_attn_grid = (img_size // 16, img_size // 16)
        s3_attn = _TransformerStage(
            D_attn,
            D_attn,
            L_attn,
            heads[0],
            input_grid=s3_attn_grid,
            downsample=False,
            dpr=[_next_dp() for _ in range(L_attn)],
        )
        s3 = nn.Sequential(*s3_mb_layers, s3_expand, s3_attn)
        s3_out_ch = D_attn
    # After S3: H/16 × W/16

    # ------------------------------------------------------------------ S4
    s4_grid = (img_size // 16, img_size // 16)  # input to S4
    s4 = _TransformerStage(
        s3_out_ch,
        d[3],
        n[3],
        heads[1],
        input_grid=s4_grid,
        dpr=[_next_dp() for _ in range(n[3])],
    )
    # After S4: H/32 × W/32

    feature_info = [
        FeatureInfo(stage=1, num_channels=d[0], reduction=4),
        FeatureInfo(stage=2, num_channels=d[1], reduction=8),
        FeatureInfo(stage=3, num_channels=s3_out_ch, reduction=16),
        FeatureInfo(stage=4, num_channels=d[3], reduction=32),
    ]
    return stem, s1, s2, s3, s4, feature_info


# ---------------------------------------------------------------------------
# CoAtNet backbone (task="base")
# ---------------------------------------------------------------------------


class CoAtNet(PretrainedModel, BackboneMixin):
    r"""CoAtNet backbone (Dai et al., 2021).

    CoAtNet is a *hybrid* backbone that interleaves depthwise
    convolutions and relative self-attention in a single four-stage
    pyramid (preceded by a two-layer convolutional stem).  The first
    two stages (:math:`S_1, S_2`) use *MBConv* blocks (squeeze-and-
    excitation, expansion ratio 4) and the last two stages
    (:math:`S_3, S_4`) use *relative-attention* transformer blocks
    operating on flattened token sequences:

    .. math::

        \mathrm{Attn}(Q, K, V)_{ij} = \mathrm{softmax}\!\left(
            \frac{Q_i K_j^\top}{\sqrt{d}} + r_{i - j}
        \right) V_j,

    where :math:`r_{i-j}` is a learned bias indexed by the *relative*
    spatial offset between tokens.  This recovers the translation
    equivariance that convolutions provide while still permitting
    global, data-dependent mixing.  Each stage downsamples
    :math:`2\times`, so the final feature map is
    :math:`(B, d_{S_4}, H/32, W/32)`.

    :meth:`forward_features` returns the raw spatial feature map from
    the last attention stage.  Use this backbone when you need
    multi-scale or spatial features for detection / segmentation; for
    end-to-end classification use
    :class:`CoAtNetForImageClassification`.

    Parameters
    ----------
    config : CoAtNetConfig
        Frozen dataclass specifying ``blocks_per_stage``, ``dims``,
        ``stem_width``, ``attn_heads``, ``mbconv_expand``,
        ``image_size``, and ``in_channels``.  See :class:`CoAtNetConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Two-layer stride-2 convolutional stem
        :math:`(3\times3 \,\mathrm{Conv}, \mathrm{BN}, \mathrm{GELU})^2`.
    s1 : nn.Sequential
        First MBConv stage with downsampling.
    s2 : nn.Sequential
        Second MBConv stage with downsampling.
    s3 : _TransformerStage
        First relative-attention transformer stage.
    s4 : _TransformerStage
        Second relative-attention transformer stage.
    feature_info : list[FeatureInfo]
        Four-stage feature description with reductions
        :math:`(4, 8, 16, 32)`.

    Notes
    -----
    Reference: Zihang Dai *et al.*, *"CoAtNet: Marrying Convolution
    and Attention for All Data Sizes"*, NeurIPS 2021,
    `arXiv:2106.04803 <https://arxiv.org/abs/2106.04803>`_.

    Examples
    --------
    Build a CoAtNet-0 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.coatnet import CoAtNet, CoAtNetConfig
    >>> model = CoAtNet(CoAtNetConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, dims[-1], H/32, W/32)
    (1, 768, 7, 7)
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

    @override
    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    @override
    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        x = cast(Tensor, self.s1(x))
        x = cast(Tensor, self.s2(x))
        x = cast(Tensor, self.s3(x))
        x = cast(Tensor, self.s4(x))
        return x

    @override
    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        return BaseModelOutput(last_hidden_state=self.forward_features(x))


# ---------------------------------------------------------------------------
# CoAtNet for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class CoAtNetForImageClassification(ImageClassificationModel, ClassificationHeadMixin):
    r"""CoAtNet with a linear classification head (Dai et al., 2021).

    Wraps the same conv + attention trunk as :class:`CoAtNet` (stem
    + two MBConv stages + two relative-attention transformer stages)
    and adds the standard reference recipe head: global average pool
    → LayerNorm → optional pre-logits Linear + Tanh → linear
    classifier.

    .. math::

        \text{logits} = W_{\text{cls}}\,
            \mathrm{Tanh}\!\bigl(W_{\text{pre}}\,
            \mathrm{LN}(\mathrm{GAP}(z^{S_4}))\bigr) + b_{\text{cls}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : CoAtNetConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  Set
        ``head_hidden_size=None`` to drop the pre-logits projection.
        See :class:`CoAtNetConfig`.

    Attributes
    ----------
    stem : nn.Sequential
        Two-layer stride-2 convolutional stem.
    s1, s2 : nn.Sequential
        Two MBConv stages.
    s3, s4 : _TransformerStage
        Two relative-attention transformer stages.
    avgpool : nn.AdaptiveAvgPool2d
        :math:`1 \times 1` adaptive average pool over spatial dims.
    norm : nn.LayerNorm
        LayerNorm applied to the pooled feature.
    pre_logits : nn.Module
        Either ``Linear + Tanh`` (when ``config.head_hidden_size`` is
        set) or an identity ``nn.Sequential``.
    classifier : nn.Linear
        Final linear projection of width ``(num_classes, head_in)``
        where ``head_in`` is either ``config.head_hidden_size`` or
        ``config.dims[-1]``.

    Notes
    -----
    Reference: Zihang Dai *et al.*, *"CoAtNet: Marrying Convolution
    and Attention for All Data Sizes"*, NeurIPS 2021.  CoAtNet-0
    reaches **81.6% top-1 on ImageNet-1k** at 224x224 (Table 5).

    Examples
    --------
    End-to-end inference with the default CoAtNet-0 classifier:

    >>> import lucid
    >>> from lucid.models.vision.coatnet import (
    ...     CoAtNetConfig, CoAtNetForImageClassification,
    ... )
    >>> model = CoAtNetForImageClassification(CoAtNetConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
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
        # Three pieces here are reference-faithful but *not* described in the
        # paper: this head LayerNorm, the pre-logits Linear+Tanh below, and
        # the per-stage LayerNorm in ``_Stage``.  They come from timm's
        # ``NormMlpClassifierHead`` (pool -> norm -> fc1 -> tanh -> fc2) and
        # from the ``head_hidden_size=768`` its paper-matching ``coatnet_0``
        # entry carries.  The cost is measurable — 24.79M for the backbone
        # against 26.15M with the head, so ~1.36M of it, versus the paper's
        # 25M total in Table 4.  Kept because the converted checkpoints have
        # these tensors; removing them would strand every shipped weight.
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

    @override
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
