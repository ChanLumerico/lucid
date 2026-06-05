"""CrossViT backbone and classifier (Chen et al., ICCV 2021).

Paper: "CrossViT: Cross-Attention Multi-Scale Vision Transformer for
Image Classification" (arXiv:2103.14899).

Architecture:
    * Two parallel ViT branches with different patch sizes.
      The *small-patch* branch processes the full image (240×240 with
      patch size 12 → 20×20 = 400 tokens); the *large-patch* branch
      processes a rescaled version (224×224 with patch size 16 →
      14×14 = 196 tokens).  Each branch carries its own learnable
      CLS token + positional embedding.
    * The trunk is K=3 ``MultiScaleBlock`` s.  Each block first runs
      ``N_s`` / ``N_l`` standard self-attention blocks on the two
      branches *independently*, then exchanges CLS tokens between the
      branches via cross-attention: the CLS from branch ``d`` is
      projected (LN + GELU + Linear) into the embedding space of
      branch ``d_=(d+1) % 2``, cross-attends to that branch's patch
      tokens, then is projected back.  This is paper §3.2.
    * Final head: per-branch ``LayerNorm`` over the CLS token + a
      per-branch ``Linear`` → ``num_classes``.  The two logits are
      averaged (paper §3.3, "ensemble of the two heads").

Implementation note: module naming mirrors timm's
``timm.models.crossvit`` so the converter only needs a single
``blocks`` → ``stages`` rename to remap a timm checkpoint.  Block-
level naming (``attn.qkv`` / ``attn.proj`` / ``mlp.fc1`` / ``mlp.fc2``)
is identical.
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
from lucid.models._utils._classification import DropPath
from lucid.models.vision.crossvit._config import CrossViTConfig

# ---------------------------------------------------------------------------
# Patch embedding (one per branch)
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    """``Conv2d(in→embed_dim, k=stride=patch)`` flattening to NLC tokens."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.proj(x))  # (B, C, H', W')
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)  # (B, N, C)


# ---------------------------------------------------------------------------
# Self-attention block (standard ViT block)
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    """Standard multi-head self-attention with fused QKV projection."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        B, N, C = x.shape
        qkv = cast(Tensor, self.qkv(x)).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.swapaxes(-2, -1)) * self.scale  # (B, H, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = cast(Tensor, self.attn_drop(attn))
        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = cast(Tensor, self.proj(x))
        return cast(Tensor, self.proj_drop(x))


class _Mlp(nn.Module):
    """Two-layer Linear-GELU-Linear MLP with dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = F.gelu(cast(Tensor, self.fc1(x)))
        x = cast(Tensor, self.drop(x))
        x = cast(Tensor, self.fc2(x))
        return cast(Tensor, self.drop(x))


class _Block(nn.Module):
    """Standard ViT transformer block: LN → MSA → LN → MLP + residuals."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = _Attention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio), dim, drop=drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        n1 = cast(Tensor, self.norm1(x))
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.attn(n1))))
        n2 = cast(Tensor, self.norm2(x))
        x = x + cast(Tensor, self.drop_path(cast(Tensor, self.mlp(n2))))
        return x


# ---------------------------------------------------------------------------
# Cross-attention block (query from one branch's CLS, KV from the other branch)
# ---------------------------------------------------------------------------


class _CrossAttention(nn.Module):
    """Cross-attention: query = single CLS token, KV = patch sequence.

    The query branch has dimension ``dim``; the KV branch has been
    pre-projected (outside this module) to the same dimension so we
    can use a single shared ``head_dim``.  Mirrors timm's
    ``CrossAttention`` exactly so the state-dict naming
    (``attn.wq`` / ``attn.wk`` / ``attn.wv`` / ``attn.proj``) carries
    over without renames.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # x = concat(cls_query, kv_sequence) along token dim.
        # timm's CrossAttention takes only the first token as query,
        # the entire sequence as KV.
        B, N, C = x.shape
        q = (
            cast(Tensor, self.wq(x[:, 0:1, :]))
            .reshape(B, 1, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        k = (
            cast(Tensor, self.wk(x))
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        v = (
            cast(Tensor, self.wv(x))
            .reshape(B, N, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        attn = (q @ k.swapaxes(-2, -1)) * self.scale  # (B, H, 1, N)
        attn = F.softmax(attn, dim=-1)
        attn = cast(Tensor, self.attn_drop(attn))
        x = (attn @ v).swapaxes(1, 2).reshape(B, 1, C)
        x = cast(Tensor, self.proj(x))
        return cast(Tensor, self.proj_drop(x))


class _CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block (no MLP, mirroring the paper)."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.attn = _CrossAttention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # Residual is added only to the CLS query token (first slot).
        cls_q = x[:, 0:1, :]
        out = cast(Tensor, self.attn(cast(Tensor, self.norm1(x))))
        return cls_q + cast(Tensor, self.drop_path(out))


# ---------------------------------------------------------------------------
# Multi-scale block: per-branch self-attention + cross-branch fusion
# ---------------------------------------------------------------------------


def _make_proj(in_dim: int, out_dim: int, layer_norm_eps: float) -> nn.Sequential:
    """LN(in) → GELU → Linear(in → out) — used for both proj and revert."""
    return nn.Sequential(
        nn.LayerNorm(in_dim, eps=layer_norm_eps),
        nn.GELU(),
        nn.Linear(in_dim, out_dim),
    )


class _MultiScaleBlock(nn.Module):
    """One CrossViT stage: independent branches → CLS-token fusion."""

    def __init__(
        self,
        dim: tuple[int, int],
        depth: tuple[int, int, int],
        num_heads: tuple[int, int],
        mlp_ratio: tuple[float, float, float],
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: tuple[list[float], list[float], list[float]] | None = None,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.num_branches = 2
        # Per-branch self-attention stacks.
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        _Block(
                            dim[d],
                            num_heads[d],
                            mlp_ratio[d],
                            qkv_bias=qkv_bias,
                            drop=drop,
                            attn_drop=attn_drop,
                            drop_path=(drop_path[d][i] if drop_path else 0.0),
                            layer_norm_eps=layer_norm_eps,
                        )
                        for i in range(depth[d])
                    ]
                )
                for d in range(self.num_branches)
            ]
        )

        # Project CLS from branch d to dim of branch d_ = (d+1) % 2.
        self.projs = nn.ModuleList(
            [
                _make_proj(dim[d], dim[(d + 1) % 2], layer_norm_eps)
                for d in range(self.num_branches)
            ]
        )

        # Cross-attention fusion: query CLS (already projected) + KV from the
        # other branch's patch tokens.  Built at the *destination* dim.
        # Paper-cited variants run exactly one cross-attention pass per
        # branch per stage (``depth[2] = 0``), so ``self.fusion[d]`` is a
        # single ``_CrossAttentionBlock`` — not a Sequential — matching
        # timm's layout exactly.
        self.fusion = nn.ModuleList(
            [
                _CrossAttentionBlock(
                    dim[(d + 1) % 2],
                    num_heads[(d + 1) % 2],
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=(drop_path[2][0] if drop_path else 0.0),
                    layer_norm_eps=layer_norm_eps,
                )
                for d in range(self.num_branches)
            ]
        )

        # Revert projection: CLS back to original branch's dim.
        self.revert_projs = nn.ModuleList(
            [
                _make_proj(dim[(d + 1) % 2], dim[d], layer_norm_eps)
                for d in range(self.num_branches)
            ]
        )

    def forward(self, xs: list[Tensor]) -> list[Tensor]:  # type: ignore[override]
        # Per-branch self-attention.
        xs = [cast(Tensor, self.blocks[d](xs[d])) for d in range(self.num_branches)]

        # Project CLS of each branch into the other branch's dim space.
        proj_cls = [
            cast(Tensor, self.projs[d](xs[d][:, 0:1, :]))
            for d in range(self.num_branches)
        ]

        # Cross-attend + revert.
        out_xs: list[Tensor] = []
        for d in range(self.num_branches):
            d_ = (d + 1) % self.num_branches
            # Concat projected CLS (1 token) + other-branch patch tokens.
            tmp = lucid.cat([proj_cls[d], xs[d_][:, 1:, :]], dim=1)
            tmp = cast(Tensor, self.fusion[d](tmp))
            # Project the fused CLS back to original branch dim and stitch
            # together with the unchanged patch tokens.
            reverted = cast(Tensor, self.revert_projs[d](tmp))
            out_xs.append(lucid.cat([reverted, xs[d][:, 1:, :]], dim=1))
        return out_xs


# ---------------------------------------------------------------------------
# Image rescaling for the dual-input forward pass.
# ---------------------------------------------------------------------------


def _bicubic_2d(x: Tensor, oh: int, ow: int) -> Tensor:
    """``align_corners=False`` 2D bicubic resampling — separable row+col.

    Mirrors the reference framework's ``F.interpolate(..., mode='bicubic',
    align_corners=False)`` exactly, with the Mitchell-Netravali kernel
    coefficient ``a = -0.75`` (reference-framework / OpenCV default).  Implemented
    in Python on top of Lucid native ops because Lucid's engine-level
    ``F.interpolate`` does not yet ship a bicubic kernel; the path
    is only hit once per forward pass in CrossViT (large-branch
    rescale), so the lack of a fused C++ kernel doesn't matter.

    Shape: ``(B, C, H, W) → (B, C, oh, ow)``.
    """
    B, C, H, W = x.shape
    a = -0.75

    def _coords(out_size: int, in_size: int) -> tuple[Tensor, Tensor]:
        src = (lucid.arange(out_size, dtype=lucid.float32) + 0.5) * (
            in_size / out_size
        ) - 0.5
        idx = src.floor().to(lucid.int64)
        frac = src - idx.to(lucid.float32)
        return idx, frac

    def _weights(
        frac: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # Distances to the four source samples are (1+t, t, 1-t, 2-t).
        d0 = 1.0 + frac
        d1 = frac
        d2 = 1.0 - frac
        d3 = 2.0 - frac
        # |d| in [1,2]: w(d) = a·d³ − 5a·d² + 8a·d − 4a
        w0 = a * (d0**3) - 5 * a * (d0**2) + 8 * a * d0 - 4 * a
        w3 = a * (d3**3) - 5 * a * (d3**2) + 8 * a * d3 - 4 * a
        # |d| in [0,1]: w(d) = (a+2)·d³ − (a+3)·d² + 1
        w1 = (a + 2) * (d1**3) - (a + 3) * (d1**2) + 1.0
        w2 = (a + 2) * (d2**3) - (a + 3) * (d2**2) + 1.0
        return w0, w1, w2, w3

    # ── Row (height) pass: (B, C, H, W) → (B, C, oh, W) ─────────────
    y_idx, y_frac = _coords(oh, H)
    wy0, wy1, wy2, wy3 = _weights(y_frac)
    # Lucid's clamp/clip backend does not yet handle int64; float
    # round-trip is the (currently) cheapest path.
    y_idx_f = y_idx.to(lucid.float32)
    y_m1 = lucid.clamp(y_idx_f - 1.0, 0.0, float(H - 1)).to(lucid.int64)
    y_p0 = lucid.clamp(y_idx_f, 0.0, float(H - 1)).to(lucid.int64)
    y_p1 = lucid.clamp(y_idx_f + 1.0, 0.0, float(H - 1)).to(lucid.int64)
    y_p2 = lucid.clamp(y_idx_f + 2.0, 0.0, float(H - 1)).to(lucid.int64)
    r_m1 = x[:, :, y_m1, :]
    r_p0 = x[:, :, y_p0, :]
    r_p1 = x[:, :, y_p1, :]
    r_p2 = x[:, :, y_p2, :]
    wy_shape = (1, 1, oh, 1)
    out_r = (
        wy0.reshape(*wy_shape) * r_m1
        + wy1.reshape(*wy_shape) * r_p0
        + wy2.reshape(*wy_shape) * r_p1
        + wy3.reshape(*wy_shape) * r_p2
    )

    # ── Column (width) pass: (B, C, oh, W) → (B, C, oh, ow) ─────────
    x_idx, x_frac = _coords(ow, W)
    wx0, wx1, wx2, wx3 = _weights(x_frac)
    x_idx_f = x_idx.to(lucid.float32)
    x_m1 = lucid.clamp(x_idx_f - 1.0, 0.0, float(W - 1)).to(lucid.int64)
    x_p0 = lucid.clamp(x_idx_f, 0.0, float(W - 1)).to(lucid.int64)
    x_p1 = lucid.clamp(x_idx_f + 1.0, 0.0, float(W - 1)).to(lucid.int64)
    x_p2 = lucid.clamp(x_idx_f + 2.0, 0.0, float(W - 1)).to(lucid.int64)
    c_m1 = out_r[:, :, :, x_m1]
    c_p0 = out_r[:, :, :, x_p0]
    c_p1 = out_r[:, :, :, x_p1]
    c_p2 = out_r[:, :, :, x_p2]
    wx_shape = (1, 1, 1, ow)
    out = (
        wx0.reshape(*wx_shape) * c_m1
        + wx1.reshape(*wx_shape) * c_p0
        + wx2.reshape(*wx_shape) * c_p1
        + wx3.reshape(*wx_shape) * c_p2
    )
    return out


def _scale_image(x: Tensor, target: int, crop_scale: bool) -> Tensor:
    """Resize (default) or center-crop the input to an absolute ``target`` side.

    Mirrors timm's ``scale_image``: the destination is a fixed pixel target
    (``round(image_size * img_scale[d])``, computed by the caller from the
    *config* — never from the runtime input dims).  This keeps each branch's
    patch count equal to its positional-embedding size for any input
    resolution.  A no-op when the input already matches ``target``.
    """
    _, _, H, W = x.shape
    if H == target and W == target:
        return x
    if crop_scale and target <= H and target <= W:
        # Center crop to ``(B, C, target, target)``.
        top = (H - target) // 2
        left = (W - target) // 2
        return x[:, :, top : top + target, left : left + target]
    # Bicubic interpolation matches timm's ``scale_image`` exactly.
    return _bicubic_2d(x, target, target)


# ---------------------------------------------------------------------------
# CrossViT backbone (task="base")
# ---------------------------------------------------------------------------


class CrossViT(PretrainedModel, BackboneMixin):
    r"""CrossViT backbone (Chen et al., ICCV 2021).

    Builds two parallel ViT branches that operate on different
    resolutions and exchange information through K=3 cross-attention
    fusion blocks.  See :class:`CrossViTConfig` for the per-variant
    hyperparameters.

    Returns
    -------
    BaseModelOutput
        ``last_hidden_state`` carries the concatenation of the two
        per-branch CLS tokens along the channel dimension.  Use
        :meth:`forward_features` to get the two CLS tokens separately.
    """

    config_class: ClassVar[type[CrossViTConfig]] = CrossViTConfig
    base_model_prefix: ClassVar[str] = "crossvit"

    def __init__(self, config: CrossViTConfig) -> None:
        super().__init__(config)
        cfg = config
        in_ch = cfg.in_channels
        eps = cfg.layer_norm_eps
        D = cfg.embed_dims
        P = cfg.patch_sizes

        # Patch embeddings for each branch.
        self.patch_embed = nn.ModuleList(
            [_PatchEmbed(in_ch, D[d], P[d]) for d in range(2)]
        )

        # CLS + positional embedding per branch.  Number of patch tokens
        # depends on per-branch input size.
        img_h = cfg.image_size
        sizes = [int(round(img_h * cfg.img_scale[d])) for d in range(2)]
        num_patches = [(sizes[d] // P[d]) ** 2 for d in range(2)]

        self.cls_token_0 = nn.Parameter(lucid.zeros(1, 1, D[0]))
        self.cls_token_1 = nn.Parameter(lucid.zeros(1, 1, D[1]))
        self.pos_embed_0 = nn.Parameter(lucid.zeros(1, num_patches[0] + 1, D[0]))
        self.pos_embed_1 = nn.Parameter(lucid.zeros(1, num_patches[1] + 1, D[1]))
        self.pos_drop = nn.Dropout(cfg.dropout)

        # Stochastic-depth schedule across all blocks in all stages.
        n_cross_total = sum(max(1, d[2] if d[2] > 0 else 1) for d in cfg.depths)
        dpr_total = sum(d[0] + d[1] for d in cfg.depths) + n_cross_total
        dpr = (
            [cfg.drop_path_rate * i / max(1, dpr_total - 1) for i in range(dpr_total)]
            if cfg.drop_path_rate > 0.0
            else [0.0] * dpr_total
        )
        cursor = 0
        stages: list[nn.Module] = []
        for stage_depth in cfg.depths:
            n_s, n_l, n_c = stage_depth
            n_c_eff = max(1, n_c if n_c > 0 else 1)
            stage_dpr = (
                dpr[cursor : cursor + n_s],
                dpr[cursor + n_s : cursor + n_s + n_l],
                dpr[cursor + n_s + n_l : cursor + n_s + n_l + n_c_eff],
            )
            stages.append(
                _MultiScaleBlock(
                    dim=D,
                    depth=stage_depth,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    drop=cfg.dropout,
                    attn_drop=0.0,
                    drop_path=stage_dpr,
                    layer_norm_eps=eps,
                )
            )
            cursor += n_s + n_l + n_c_eff
        self.stages = nn.ModuleList(stages)

        # Final per-branch LayerNorm.
        self.norm = nn.ModuleList([nn.LayerNorm(D[d], eps=eps) for d in range(2)])

        self._feature_info = [
            FeatureInfo(stage=1, num_channels=D[0], reduction=P[0]),
            FeatureInfo(stage=2, num_channels=D[1], reduction=P[1]),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def _branch_tokens(self, x: Tensor, branch: int) -> Tensor:
        """Patch-embed branch + CLS + positional embedding + dropout."""
        x = cast(Tensor, self.patch_embed[branch](x))  # (B, N, C)
        cls_token = (self.cls_token_0 if branch == 0 else self.cls_token_1).expand(
            x.shape[0], 1, x.shape[2]
        )
        x = lucid.cat([cls_token, x], dim=1)
        pos = self.pos_embed_0 if branch == 0 else self.pos_embed_1
        x = x + pos
        return cast(Tensor, self.pos_drop(x))

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:  # type: ignore[override]
        cfg = cast(CrossViTConfig, self.config)
        # Rescale per branch to the same absolute targets used to size the
        # positional embeddings in ``__init__`` (round(image_size·img_scale)).
        targets = [int(round(cfg.image_size * cfg.img_scale[d])) for d in range(2)]
        xs: list[Tensor] = [
            _scale_image(x, targets[d], cfg.crop_scale) for d in range(2)
        ]
        # Patch-embed + CLS + positional.
        xs = [self._branch_tokens(xs[d], d) for d in range(2)]
        # K cross-attention stages — _MultiScaleBlock takes/returns ``list[Tensor]``.
        for stage in self.stages:
            xs = stage(xs)  # type: ignore[arg-type,assignment]
        # Final norm per branch.
        xs = [cast(Tensor, self.norm[d](xs[d])) for d in range(2)]
        # Return the two CLS tokens.
        return xs[0][:, 0], xs[1][:, 0]

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        cls_s, cls_l = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=lucid.cat([cls_s, cls_l], dim=-1))


# ---------------------------------------------------------------------------
# CrossViT for image classification (task="image-classification")
# ---------------------------------------------------------------------------


class CrossViTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""CrossViT with two per-branch linear heads, output averaged.

    Paper §3.3 — the final prediction is the *mean* of the two
    branch-specific logits, an implicit two-model ensemble.
    """

    config_class: ClassVar[type[CrossViTConfig]] = CrossViTConfig
    base_model_prefix: ClassVar[str] = "crossvit"

    def __init__(self, config: CrossViTConfig) -> None:
        super().__init__(config)
        cfg = config
        D = cfg.embed_dims
        P = cfg.patch_sizes
        eps = cfg.layer_norm_eps
        in_ch = cfg.in_channels

        # Trunk (mirrors :class:`CrossViT` exactly so the state-dict
        # layout is shared and a converter can target either head).
        self.patch_embed = nn.ModuleList(
            [_PatchEmbed(in_ch, D[d], P[d]) for d in range(2)]
        )
        img_h = cfg.image_size
        sizes = [int(round(img_h * cfg.img_scale[d])) for d in range(2)]
        num_patches = [(sizes[d] // P[d]) ** 2 for d in range(2)]
        self.cls_token_0 = nn.Parameter(lucid.zeros(1, 1, D[0]))
        self.cls_token_1 = nn.Parameter(lucid.zeros(1, 1, D[1]))
        self.pos_embed_0 = nn.Parameter(lucid.zeros(1, num_patches[0] + 1, D[0]))
        self.pos_embed_1 = nn.Parameter(lucid.zeros(1, num_patches[1] + 1, D[1]))
        self.pos_drop = nn.Dropout(cfg.dropout)

        n_cross_total = sum(max(1, d[2] if d[2] > 0 else 1) for d in cfg.depths)
        dpr_total = sum(d[0] + d[1] for d in cfg.depths) + n_cross_total
        dpr = (
            [cfg.drop_path_rate * i / max(1, dpr_total - 1) for i in range(dpr_total)]
            if cfg.drop_path_rate > 0.0
            else [0.0] * dpr_total
        )
        cursor = 0
        stages: list[nn.Module] = []
        for stage_depth in cfg.depths:
            n_s, n_l, n_c = stage_depth
            n_c_eff = max(1, n_c if n_c > 0 else 1)
            stage_dpr = (
                dpr[cursor : cursor + n_s],
                dpr[cursor + n_s : cursor + n_s + n_l],
                dpr[cursor + n_s + n_l : cursor + n_s + n_l + n_c_eff],
            )
            stages.append(
                _MultiScaleBlock(
                    dim=D,
                    depth=stage_depth,
                    num_heads=cfg.num_heads,
                    mlp_ratio=cfg.mlp_ratio,
                    qkv_bias=cfg.qkv_bias,
                    drop=cfg.dropout,
                    attn_drop=0.0,
                    drop_path=stage_dpr,
                    layer_norm_eps=eps,
                )
            )
            cursor += n_s + n_l + n_c_eff
        self.stages = nn.ModuleList(stages)
        self.norm = nn.ModuleList([nn.LayerNorm(D[d], eps=eps) for d in range(2)])

        # Per-branch classifier heads (averaged at the logit level).
        self.head = nn.ModuleList([nn.Linear(D[d], cfg.num_classes) for d in range(2)])

    def _branch_tokens(self, x: Tensor, branch: int) -> Tensor:
        x = cast(Tensor, self.patch_embed[branch](x))
        cls_token = (self.cls_token_0 if branch == 0 else self.cls_token_1).expand(
            x.shape[0], 1, x.shape[2]
        )
        x = lucid.cat([cls_token, x], dim=1)
        pos = self.pos_embed_0 if branch == 0 else self.pos_embed_1
        x = x + pos
        return cast(Tensor, self.pos_drop(x))

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        cfg = cast(CrossViTConfig, self.config)
        targets = [int(round(cfg.image_size * cfg.img_scale[d])) for d in range(2)]
        xs: list[Tensor] = [
            _scale_image(x, targets[d], cfg.crop_scale) for d in range(2)
        ]
        xs = [self._branch_tokens(xs[d], d) for d in range(2)]
        for stage in self.stages:
            xs = stage(xs)  # type: ignore[arg-type,assignment]
        xs = [cast(Tensor, self.norm[d](xs[d])) for d in range(2)]
        cls = [xs[d][:, 0] for d in range(2)]
        logits_s = cast(Tensor, self.head[0](cls[0]))
        logits_l = cast(Tensor, self.head[1](cls[1]))
        logits = (logits_s + logits_l) / 2.0

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return ImageClassificationOutput(logits=logits, loss=loss)
