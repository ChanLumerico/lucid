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
  Stem   : Conv3×3(s=2, 3→64, bias) → BN → GELU → Conv3×3(s=1, 64→64, bias)
  Stage 0: 2 × MaxViTBlock(64, stride1=2, stride_rest=1) → (56×56)
  Stage 1: 2 × MaxViTBlock(128, stride1=2, stride_rest=1) → (28×28)
  Stage 2: 5 × MaxViTBlock(256, stride1=2, stride_rest=1) → (14×14)
  Stage 3: 2 × MaxViTBlock(512, stride1=2, stride_rest=1) → (7×7)
  Head   : LayerNorm → Pre-logits Linear → GELU (or Tanh) → FC

Key attribute naming mirrors timm's maxvit_tiny_tf_224 exactly so that
state-dict transfer works without any key remapping.

MBConv per block:
  conv.pre_norm (BN) → conv.conv1_1x1 → conv.norm1 (BN+GELU) →
  conv.conv2_kxk (DW, stride) → conv.norm2 (BN+GELU) →
  conv.se (SE: fc1/fc2 as Conv2d) →
  conv.conv3_1x1 (projection, no BN/act, with bias)

  Shortcut: when in_dim != out_dim, conv.shortcut.expand (1×1 Conv2d).

Attention blocks:
  attn_block.norm1 / .attn.qkv / .attn.rel_pos / .attn.proj / .norm2 / .mlp
  attn_grid.norm1  / .attn.qkv / .attn.rel_pos / .attn.proj / .norm2 / .mlp

  rel_pos holds a learnable table (num_heads, 2*ws-1, 2*ws-1) — RelPosBias.
  head_dim = 32 (fixed); num_heads = dim // 32.

Head:
  head.norm (LayerNorm) → head.pre_logits.fc (Linear) → Tanh →
  head.fc (Linear, num_classes)
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
from lucid.models.vision.maxvit._config import MaxViTConfig

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HEAD_DIM: int = 32  # fixed head dimension (same as timm default)
_SE_RATIO: int = 16  # SE squeeze ratio relative to expanded dim

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
    """(B, H, W, C) → (B*nH*nW, ws, ws, C).

    Matches timm's grid_partition(x, grid_size=(ws, ws)).

    Each of the B*nH*nW groups contains the ws×ws pixels sampled at the
    same grid-stride offset.  Reshape:
      x.view(B, ws, nH, ws, nW, C) → permute(0,2,4,1,3,5)
      → view(-1, ws, ws, C)

    Requires H, W divisible by ws (caller must pad if needed).
    """
    B, H, W, C = x.shape
    nH, nW = H // ws, W // ws
    x = x.reshape(B, ws, nH, ws, nW, C)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, nH, nW, ws, ws, C)
    x = x.reshape(B * nH * nW, ws, ws, C)
    return x, nH, nW


def _grid_reverse(x: Tensor, ws: int, nH: int, nW: int, B: int) -> Tensor:
    """(B*nH*nW, ws, ws, C) → (B, H, W, C)."""
    C = x.shape[-1]
    x = x.reshape(B, nH, nW, ws, ws, C)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (B, ws, nH, ws, nW, C)
    return x.reshape(B, ws * nH, ws * nW, C)


# ---------------------------------------------------------------------------
# Helpers: pad spatial to multiple of ws, then crop back
# ---------------------------------------------------------------------------


def _tf_same_pad2d(x: Tensor, kernel_size: int, stride: int) -> Tensor:
    """Apply TF-style 'same' padding to a NCHW tensor.

    timm uses ``Conv2dSame`` for stride-2 convolutions in MaxViT.  TF pads
    right/bottom first (ceil division), giving asymmetric padding whenever
    the total pad is odd.  Standard ``padding=1`` in most frameworks pads
    symmetrically, which produces different border values for real inputs.
    """
    H, W = x.shape[2], x.shape[3]
    pad_h = max(int(math.ceil(H / stride) - 1) * stride + kernel_size - H, 0)
    pad_w = max(int(math.ceil(W / stride) - 1) * stride + kernel_size - W, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


def _pad_to_multiple(x_cl: Tensor, ws: int) -> tuple[Tensor, int, int]:
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
    return x_cl, H, W


# ---------------------------------------------------------------------------
# Relative-position bias (matches timm's RelPosBiasTf)
# ---------------------------------------------------------------------------


class _RelPosBias(nn.Module):
    """Learnable relative position bias table: (num_heads, 2*ws-1, 2*ws-1).

    Matches timm's ``RelPosBiasTf`` — the table is indexed by (row_offset,
    col_offset) in [-ws+1, ws-1] range.  We store as ``relative_position_bias_table``
    so the key name matches timm exactly.

    At forward, we gather from the table using pre-built Python index lists so
    we don't need fancy integer-tensor indexing in Lucid.
    """

    def __init__(self, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.ws = window_size
        self.num_heads = num_heads
        table_size = 2 * window_size - 1
        self.relative_position_bias_table = nn.Parameter(
            lucid.zeros(num_heads, table_size, table_size)
        )
        # Build flat Python index lists for the gather at forward time.
        # coords: list of (row, col) for each of the ws² positions
        coords = [(i, j) for i in range(window_size) for j in range(window_size)]
        n = len(coords)
        # row_flat[k] and col_flat[k] are the table row/col index for
        # the k-th entry in the flattened (n*n,) gather.
        row_flat: list[int] = []
        col_flat: list[int] = []
        for ci in coords:
            for cj in coords:
                row_flat.append(ci[0] - cj[0] + (window_size - 1))
                col_flat.append(ci[1] - cj[1] + (window_size - 1))
        self._n = n
        self._row_flat = row_flat
        self._col_flat = col_flat

    def forward(self) -> Tensor:  # type: ignore[override]
        """Return bias tensor of shape (1, num_heads, ws², ws²)."""
        table = self.relative_position_bias_table  # (H, T, T)
        n = self._n
        H = self.num_heads
        # Gather: for each (i,j) pair, pick table[h, row, col]
        # Build (H, n*n) bias by stacking slices
        flat_len = len(self._row_flat)
        # Collect column slices from table for each pair position
        # table shape: (H, T, T); we gather flat_len entries
        bias_list: list[Tensor] = []
        for k in range(flat_len):
            r, c = self._row_flat[k], self._col_flat[k]
            bias_list.append(table[:, r, c])  # (H,)
        # Stack to (flat_len, H) then transpose to (H, flat_len)
        bias_flat = lucid.stack(bias_list, dim=1)  # (H, flat_len)
        bias = bias_flat.reshape(H, n, n)  # (H, ws², ws²)
        return bias.unsqueeze(0)  # (1, H, ws², ws²)


# ---------------------------------------------------------------------------
# Scaled dot-product attention with relative position bias
# (channel-last, used by both block and grid attention)
# ---------------------------------------------------------------------------


class _AttnCl(nn.Module):
    """Attention with relative position bias.

    Matches timm's ``AttentionCl`` structure:
      qkv: Linear(dim, 3*dim)
      rel_pos: RelPosBiasTf  (→ relative_position_bias_table)
      proj: Linear(dim, dim)

    Input/output: (*, N, C) channel-last format.
    """

    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim) ** -1
        self.qkv = nn.Linear(dim, dim * 3)
        self.rel_pos = _RelPosBias(num_heads, window_size)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, N, C)  where B may be a compound batch
        B, N, C = x.shape
        H = self.num_heads
        hd = self.head_dim

        qkv = cast(Tensor, self.qkv(x))  # (B, N, 3C)
        qkv = qkv.reshape(B, N, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, N, hd)

        attn = q @ k.permute(0, 1, 3, 2) * self.scale  # (B, H, N, N)

        # Add relative position bias: (1, H, N, N)
        bias = cast(Tensor, self.rel_pos())
        attn = attn + bias

        attn = F.softmax(attn, dim=-1)  # (B, H, N, N)

        out = attn @ v  # (B, H, N, hd)
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return cast(Tensor, self.proj(out))


# ---------------------------------------------------------------------------
# MLP block (matches timm's Mlp with fc1/fc2 named children)
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    """Two-layer MLP with GELU: fc1 → GELU → fc2.

    Matches timm Mlp attribute names so keys transfer directly.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.fc1(x))
        x = F.gelu(x)
        return cast(Tensor, self.fc2(x))


# ---------------------------------------------------------------------------
# Partition attention block (Pre-LN, matches timm's PartitionAttentionCl)
# ---------------------------------------------------------------------------


class _PartitionAttn(nn.Module):
    """Pre-LN attention block matching timm's PartitionAttentionCl.

    Attribute layout:
      norm1 / attn / norm2 / mlp
    """

    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: float, window_size: int
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _AttnCl(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = _MLP(dim, hidden)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x + cast(Tensor, self.attn(cast(Tensor, self.norm1(x))))
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# SE (Squeeze-and-Excitation) sub-module using Conv2d fc1/fc2
# Matches timm's SEModule with fc1/fc2 as Conv2d
# ---------------------------------------------------------------------------


class _SE(nn.Module):
    """Squeeze-and-Excitation with Conv2d fc1 and fc2 (matching timm SEModule).

    fc1: Conv2d(expanded_dim, se_mid, 1)
    fc2: Conv2d(se_mid, expanded_dim, 1)
    """

    def __init__(self, expanded_dim: int) -> None:
        super().__init__()
        se_mid = max(1, expanded_dim // _SE_RATIO)
        self.fc1 = nn.Conv2d(expanded_dim, se_mid, 1)
        self.fc2 = nn.Conv2d(se_mid, expanded_dim, 1)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C, H, W)
        s = F.adaptive_avg_pool2d(x, (1, 1))
        s = F.silu(cast(Tensor, self.fc1(s)))
        s = F.sigmoid(cast(Tensor, self.fc2(s)))
        return x * s


# ---------------------------------------------------------------------------
# Shortcut module (matches timm's Downsample2d for cross-stage projection)
# ---------------------------------------------------------------------------


class _Shortcut(nn.Module):
    """Cross-stage shortcut: AvgPool (stride) + expand Conv2d.

    Matches timm's Downsample2d with pool + expand:
      - stride > 1: pool = AvgPool2d(stride)
      - in_dim != out_dim: expand = Conv2d(in_dim, out_dim, 1, bias=True)
      - in_dim == out_dim: expand = Identity (no state-dict keys)

    Stage 0 block 0 has stride=2, same dim → pool only (expand is Identity).
    Cross-stage blocks have stride=2, different dims → pool + Conv2d expand.
    """

    def __init__(self, in_dim: int, out_dim: int, stride: int) -> None:
        super().__init__()
        if stride > 1:
            self.pool: nn.Module = nn.AvgPool2d(stride, stride=stride)
        else:
            self.pool = nn.Identity()
        if in_dim != out_dim:
            self.expand: nn.Module = nn.Conv2d(in_dim, out_dim, 1, bias=True)
        else:
            self.expand = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.pool(x))
        return cast(Tensor, self.expand(x))


# ---------------------------------------------------------------------------
# MBConv block (matches timm's MbConvBlock)
# ---------------------------------------------------------------------------


class _MBConv(nn.Module):
    """MBConv: BN pre-norm → expand Conv1×1 → BN+GELU →
    DWConv3×3(stride) → BN+GELU → SE → project Conv1×1 (bias).

    Shortcut: identity or Downsample2d (when dims differ / stride > 1).

    Attribute names mirror timm's MbConvBlock exactly:
      shortcut (optional, only when in_dim != out_dim),
      pre_norm, conv1_1x1, norm1, conv2_kxk, norm2, se, conv3_1x1
    """

    def __init__(
        self, in_dim: int, out_dim: int, stride: int = 1, expand_ratio: int = 4
    ) -> None:
        super().__init__()
        mid = out_dim * expand_ratio

        # Shortcut: always create when in_dim != out_dim or stride > 1
        if in_dim != out_dim or stride > 1:
            self.shortcut = _Shortcut(in_dim, out_dim, stride)
        # If same dim and stride=1, no shortcut attribute (identity skip)

        self.pre_norm = nn.BatchNorm2d(in_dim)
        self.conv1_1x1 = nn.Conv2d(in_dim, mid, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(mid)
        # DW conv with stride; when stride=2 timm uses Conv2dSame (TF-same padding),
        # so we store padding=0 and apply _tf_same_pad2d manually in forward.
        self._conv2_stride = stride
        self.conv2_kxk = nn.Conv2d(
            mid,
            mid,
            3,
            stride=stride,
            padding=0 if stride > 1 else 1,
            groups=mid,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(mid)
        self.se = _SE(mid)
        self.conv3_1x1 = nn.Conv2d(mid, out_dim, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Shortcut path
        if hasattr(self, "shortcut"):
            shortcut = cast(Tensor, self.shortcut(x))
        else:
            shortcut = x

        # Main path
        out = cast(Tensor, self.pre_norm(x))
        out = cast(Tensor, self.conv1_1x1(out))
        out = F.gelu(cast(Tensor, self.norm1(out)))
        if self._conv2_stride > 1:
            out = _tf_same_pad2d(out, kernel_size=3, stride=self._conv2_stride)
        out = cast(Tensor, self.conv2_kxk(out))
        out = F.gelu(cast(Tensor, self.norm2(out)))
        out = cast(Tensor, self.se(out))
        out = cast(Tensor, self.conv3_1x1(out))
        return shortcut + out


# ---------------------------------------------------------------------------
# MaxViT block: conv (MBConv) → attn_block (window) → attn_grid (grid)
# ---------------------------------------------------------------------------


class _MaxViTBlock(nn.Module):
    """Single MaxViT block = MBConv + window attention + grid attention.

    Attribute naming matches timm's MaxxVitBlock:
      conv, attn_block, attn_grid

    The stride/in_dim/out_dim for conv come from the stage configuration.
    Window and grid attention convert to channel-last internally.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.ws = window_size
        self.conv = _MBConv(in_dim, out_dim, stride=stride)
        self.attn_block = _PartitionAttn(out_dim, num_heads, mlp_ratio, window_size)
        self.attn_grid = _PartitionAttn(out_dim, num_heads, mlp_ratio, window_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C_in, H, W)
        ws = self.ws

        # 1. MBConv (NCHW)
        x = cast(Tensor, self.conv(x))

        B, C, H, W = x.shape
        x_cl = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Pad to multiple of ws
        x_cl, orig_H, orig_W = _pad_to_multiple(x_cl, ws)
        Hp, Wp = x_cl.shape[1], x_cl.shape[2]

        # 2. Block (window) attention — local within ws×ws windows
        wins, nH, nW = _window_partition(x_cl, ws)  # (B*nH*nW, ws, ws, C)
        wins_seq = wins.reshape(-1, ws * ws, C)
        wins_seq = cast(Tensor, self.attn_block(wins_seq))
        wins = wins_seq.reshape(-1, ws, ws, C)
        x_cl = _window_reverse(wins, ws, nH, nW)  # (B, Hp, Wp, C)

        # 3. Grid attention — (B*nH*nW, ws², C) like window attention
        grids, gH, gW = _grid_partition(x_cl, ws)  # (B*nH*nW, ws, ws, C)
        grids_seq = grids.reshape(-1, ws * ws, C)  # (B*nH*nW, ws², C)
        grids_seq = cast(Tensor, self.attn_grid(grids_seq))
        grids = grids_seq.reshape(-1, ws, ws, C)
        x_cl = _grid_reverse(grids, ws, gH, gW, B)  # (B, Hp, Wp, C)

        # Crop back to (post-stride) spatial size
        x_cl = x_cl[:, :orig_H, :orig_W, :]

        return x_cl.permute(0, 3, 1, 2)  # (B, C, H, W)


# ---------------------------------------------------------------------------
# Stage: nn.Sequential of blocks inside a 'blocks' attribute
# (matches timm's MaxxVitStage.blocks)
# ---------------------------------------------------------------------------


class _MaxViTStage(nn.Module):
    """One MaxViT stage: a 'blocks' Sequential of _MaxViTBlock modules.

    Matches timm's MaxxVitStage which exposes a 'blocks' Sequential child.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        block_list: list[nn.Module] = []
        for i in range(depth):
            # First block does spatial downsampling (stride=2) and dim expansion
            blk_in = in_dim if i == 0 else out_dim
            blk_stride = 2 if i == 0 else 1
            block_list.append(
                _MaxViTBlock(
                    in_dim=blk_in,
                    out_dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    stride=blk_stride,
                )
            )
        self.blocks = nn.Sequential(*block_list)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return cast(Tensor, self.blocks(x))


# ---------------------------------------------------------------------------
# Head sub-modules matching timm's NormMlpClassifierHead structure
# ---------------------------------------------------------------------------


class _PreLogits(nn.Module):
    """Pre-logits: Linear → Tanh  (matches timm head.pre_logits.fc)."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        return F.tanh(cast(Tensor, self.fc(x)))


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_maxvit(cfg: MaxViTConfig) -> tuple[
    nn.Module,
    nn.ModuleList,
    list[FeatureInfo],
    int,
]:
    """Build shared MaxViT body: stem + stages.

    Stem: Conv3×3(s=2, bias) → BN → GELU → Conv3×3(s=1, bias)
    (attribute names: stem.conv1, stem.norm1, stem.conv2)

    Stages: stages.N = _MaxViTStage with internal 'blocks' Sequential
    """
    stem_out = cfg.dims[0]

    class _Stem(nn.Module):
        """Two-conv stem matching timm's stem.conv1 / stem.norm1 / stem.conv2."""

        def __init__(self, in_ch: int, out_ch: int) -> None:
            super().__init__()
            # padding=0: TF-same is applied manually in forward before conv1
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=0, bias=True)
            self.norm1 = nn.BatchNorm2d(out_ch)
            # conv2 is stride=1 so symmetric padding=1 matches TF-same exactly
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=True)

        def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
            x = _tf_same_pad2d(x, kernel_size=3, stride=2)
            x = cast(Tensor, self.conv1(x))
            x = F.gelu(cast(Tensor, self.norm1(x)))
            return cast(Tensor, self.conv2(x))

    stem: nn.Module = _Stem(cfg.in_channels, stem_out)

    stages_list: list[nn.Module] = []
    fi: list[FeatureInfo] = []
    reduction = (
        4  # stem stride=2, then each stage block0 stride=2 → 4 total after stage 0
    )

    prev_dim = cfg.dims[0]
    for i, (depth, dim) in enumerate(zip(cfg.depths, cfg.dims)):
        num_heads = dim // _HEAD_DIM
        stages_list.append(
            _MaxViTStage(
                in_dim=prev_dim,
                out_dim=dim,
                depth=depth,
                num_heads=num_heads,
                window_size=cfg.window_size,
                mlp_ratio=cfg.mlp_ratio,
            )
        )
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=reduction))
        prev_dim = dim
        reduction *= 2

    return stem, nn.ModuleList(stages_list), fi, cfg.dims[-1]


# ---------------------------------------------------------------------------
# MaxViT backbone
# ---------------------------------------------------------------------------


class MaxViT(PretrainedModel, BackboneMixin):
    """MaxViT feature extractor — global avg-pooled final stage features."""

    config_class: ClassVar[type[MaxViTConfig]] = MaxViTConfig
    base_model_prefix: ClassVar[str] = "maxvit"

    def __init__(self, config: MaxViTConfig) -> None:
        super().__init__(config)
        stem, stages, fi, out_dim = _build_maxvit(config)
        self.stem = stem
        self.stages = stages
        self._feature_info = fi
        self._out_dim = out_dim
        self.head = _HeadNorm(out_dim, out_dim, config.num_classes)

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        x = cast(Tensor, self.stem(x))
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        # Global average pool → flatten
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        return x

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# Head module (matches timm NormMlpClassifierHead)
# ---------------------------------------------------------------------------


class _HeadNorm(nn.Module):
    """Classifier head: LayerNorm2d → flatten → pre_logits.fc → Tanh → fc.

    Attribute names match timm's NormMlpClassifierHead:
      head.norm, head.pre_logits.fc, head.fc
    """

    def __init__(self, in_dim: int, pre_logits_dim: int, num_classes: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.pre_logits = _PreLogits(in_dim, pre_logits_dim)
        self.fc = nn.Linear(pre_logits_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x: (B, C) after global pool + flatten
        x = cast(Tensor, self.norm(x))
        x = cast(Tensor, self.pre_logits(x))
        return cast(Tensor, self.fc(x))


# ---------------------------------------------------------------------------
# MaxViT for image classification
# ---------------------------------------------------------------------------


class MaxViTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    """MaxViT with global avg-pool + FC classifier head.

    State-dict keys match timm's maxvit_tiny_tf_224 exactly:
      stem.conv1/norm1/conv2, stages.N.blocks.M.conv.*, stages.N.blocks.M.attn_block.*,
      stages.N.blocks.M.attn_grid.*, head.norm, head.pre_logits.fc, head.fc
    """

    config_class: ClassVar[type[MaxViTConfig]] = MaxViTConfig
    base_model_prefix: ClassVar[str] = "maxvit"

    def __init__(self, config: MaxViTConfig) -> None:
        super().__init__(config)
        stem, stages, _, out_dim = _build_maxvit(config)
        self.stem = stem
        self.stages = stages
        self.head = _HeadNorm(out_dim, out_dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        x = cast(Tensor, self.stem(x))
        for stage in self.stages:
            x = cast(Tensor, stage(x))
        # Global average pool → flatten → LayerNorm expects (B, C)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        logits = cast(Tensor, self.head(x))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
