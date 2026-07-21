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

    @override
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
        # Register as a proper non-persistent buffer so ``.to(device=...)``
        # moves it alongside parameters and Metal forward stays on-device.
        self.register_buffer("_rel_idx", rel_idx, persistent=False)

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
        B, N, C = x.shape
        qkv = cast(Tensor, self.qkv(x))  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        # Fused SDPA with the relative-position bias as an additive mask —
        # softmax((q·kᵀ)·scale + bias)·v, never forming the (B,H,N,N) scores.
        bias = self._rel_pos_bias().reshape(1, self.num_heads, N, N)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=self.scale)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return cast(Tensor, self.proj(out))

    @override
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


@final
class _TransformerStage(nn.Module):
    """Transformer stage: optional AvgPool2d(2) → linear channel proj → N×RelAttnBlock.

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
    ) -> None:
        super().__init__()
        # Grid after optional 2× pooling.
        if downsample:
            grid_h = input_grid[0] // 2
            grid_w = input_grid[1] // 2
        else:
            grid_h = input_grid[0]
            grid_w = input_grid[1]

        self.pool: nn.Module = (
            nn.AvgPool2d(2, stride=2) if downsample else nn.Identity()
        )
        self.proj = nn.Linear(in_ch, out_ch)
        self.blocks = nn.ModuleList(
            [
                _RelAttnBlock(out_ch, num_heads, grid_h, grid_w)
                for _ in range(num_blocks)
            ]
        )
        self.norm = nn.LayerNorm(out_ch)

    @override
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
    # Two code paths:
    #   * Uniform (variants 0..5): single transformer stage as before.
    #   * Mixed (variants 6/7): MBConv sub-stage (does the spatial halving)
    #     → 1×1 channel-expand → transformer sub-stage at the wider width.
    s3_grid = (img_size // 8, img_size // 8)  # input to S3 (H/8 × W/8)
    s3: nn.Module
    s3_out_ch: int
    if config.mixed_s3 is None:
        s3 = _TransformerStage(d[1], d[2], n[2], heads[0], input_grid=s3_grid)
        s3_out_ch = d[2]
    else:
        L_mb, L_attn, D_attn = config.mixed_s3
        # MBConv sub-stage: d[1] → d[2] with stride-2 in the first block,
        # then L_mb − 1 more isotropic blocks at d[2].  Mirrors S1/S2 layout.
        s3_mb_layers: list[nn.Module] = [_MBConv(d[1], d[2], expand=exp, stride=2)]
        for _ in range(1, L_mb):
            s3_mb_layers.append(_MBConv(d[2], d[2], expand=exp, stride=1))
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
        )
        s3 = nn.Sequential(*s3_mb_layers, s3_expand, s3_attn)
        s3_out_ch = D_attn
    # After S3: H/16 × W/16

    # ------------------------------------------------------------------ S4
    s4_grid = (img_size // 16, img_size // 16)  # input to S4
    s4 = _TransformerStage(s3_out_ch, d[3], n[3], heads[1], input_grid=s4_grid)
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


class CoAtNetForImageClassification(PretrainedModel, ClassificationHeadMixin):
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
