"""CvT backbone and classification head (Wu et al., 2021).

Paper: "CvT: Introducing Convolutions to Vision Transformers"
        https://arxiv.org/abs/2103.15808

Key idea: replace standard linear Q/K/V projections in MHA with
depthwise-separable convolutional projections (DWConv3×3 + BN → flatten →
Linear). This gives translation equivariance + local context to the attention
queries/keys/values, capturing both local and global relationships.

Unlike plain ViT there are NO positional embeddings — position information
is implicitly encoded by the strided convolutional token embeddings.

Architecture (CvT-13, image=224):
  Stage 1: ConvEmbed(s=4) → (56×56, 64),   1 × CvTBlock(heads=1)
  Stage 2: ConvEmbed(s=2) → (28×28, 192),  2 × CvTBlock(heads=3)
  Stage 3: ConvEmbed(s=2) → (14×14, 384), 10 × CvTBlock(heads=6)
  Head   : LN → mean-pool → FC(384, num_classes)
"""

from typing import ClassVar, cast

import lucid.nn as nn
import lucid.nn.functional as F
from lucid._tensor.tensor import Tensor
from lucid.models._base import PretrainedModel
from lucid.models._mixins import BackboneMixin, ClassificationHeadMixin, FeatureInfo
from lucid.models._output import BaseModelOutput, ImageClassificationOutput
from lucid.models.vision.cvt._config import CvTConfig

# ---------------------------------------------------------------------------
# Convolutional token embedding (overlapping)
# ---------------------------------------------------------------------------


class _ConvEmbed(nn.Module):
    """Overlapping convolutional patch embedding.

    Conv2d (kernel > stride, so overlapping) → LN → return spatial map.
    No explicit positional embeddings needed since conv is position-aware.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int,
        stride: int,
        padding: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:  # type: ignore[override]
        # (B, C_in, H, W) → (B, C_out, H', W')
        x = cast(Tensor, self.proj(x))
        B, C, H, W = x.shape
        # (B, H'*W', C) for LayerNorm, then back
        x_flat = x.reshape(B, C, H * W).permute(0, 2, 1)
        x_flat = cast(Tensor, self.norm(x_flat))
        # Return spatial layout for downstream conv projections
        x_out = x_flat.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out, H, W


# ---------------------------------------------------------------------------
# Convolutional Q/K/V projection (the CvT innovation)
# ---------------------------------------------------------------------------


class _ConvProj(nn.Module):
    """Depthwise conv (3×3, stride) + BN + flatten → linear projection.

    For K and V: stride_kv may downsample the spatial map.
    For Q: stride=1 (no downsampling, full resolution).
    """

    def __init__(
        self,
        dim: int,
        kernel: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__()
        # Depthwise conv (groups=dim) + BN
        self.dw = nn.Conv2d(
            dim, dim, kernel, stride=stride, padding=padding, groups=dim
        )
        self.bn = nn.BatchNorm2d(dim)
        # Linear projection (bias=False as in the paper)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        # x: (B, N, C) — reshape to spatial for DWConv
        B, N, C = x.shape
        x_2d = x.permute(0, 2, 1).reshape(B, C, H, W)
        x_2d = cast(Tensor, self.bn(cast(Tensor, self.dw(x_2d))))  # (B, C, H', W')
        _B, _C, H2, W2 = x_2d.shape
        x_flat = x_2d.reshape(B, C, H2 * W2).permute(0, 2, 1)  # (B, H'*W', C)
        return cast(Tensor, self.proj(x_flat))


# ---------------------------------------------------------------------------
# CvT Attention: convolutional projections for Q, K, V
# ---------------------------------------------------------------------------


class _CvTAttention(nn.Module):
    """Multi-head self-attention with convolutional Q/K/V projections.

    Q is projected with stride=1 (full resolution).
    K and V are projected with stride=stride_kv (can downsample spatially).
    This reduces the O(N²) cost at high resolutions (like SRA in PVT).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride_kv: int = 1,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Q: stride=1 (full resolution)
        self.proj_q = _ConvProj(dim, kernel=3, stride=1, padding=1)
        # K, V: strided (may reduce spatial resolution)
        self.proj_k = _ConvProj(dim, kernel=3, stride=stride_kv, padding=1)
        self.proj_v = _ConvProj(dim, kernel=3, stride=stride_kv, padding=1)

        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        B, N, C = x.shape
        head_dim = C // self.num_heads

        q = cast(Tensor, self.proj_q(x, H, W))  # type: ignore[arg-type]  # (B, N, C)
        k = cast(Tensor, self.proj_k(x, H, W))  # type: ignore[arg-type]  # (B, N', C)
        v = cast(Tensor, self.proj_v(x, H, W))  # type: ignore[arg-type]  # (B, N', C)

        Nq = q.shape[1]
        Nkv = k.shape[1]

        q = q.reshape(B, Nq, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, Nkv, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, Nkv, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.permute(0, 1, 3, 2)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, Nq, C)
        return cast(Tensor, self.out_proj(out))


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.drop(F.gelu(cast(Tensor, self.fc1(x)))))
        return cast(Tensor, self.drop(cast(Tensor, self.fc2(x))))


# ---------------------------------------------------------------------------
# CvT transformer block
# ---------------------------------------------------------------------------


class _CvTBlock(nn.Module):
    """Pre-norm CvT block: LN → ConvAttn → residual → LN → MLP → residual."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        stride_kv: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _CvTAttention(dim, num_heads, stride_kv=stride_kv)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, mlp_ratio, dropout)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:  # type: ignore[override]
        # x: (B, N, C)
        n = cast(Tensor, self.norm1(x))
        x = x + cast(Tensor, self.attn(n, H, W))  # type: ignore[arg-type]
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# CvT stage
# ---------------------------------------------------------------------------


class _CvTStage(nn.Module):
    """One CvT stage = ConvEmbed + N × CvTBlock.

    Input: (B, C_in, H_in, W_in)  [except stage 0 which also accepts this]
    Output: (B, C_out, H_out, W_out) spatial map
    """

    def __init__(
        self,
        in_ch: int,
        dim: int,
        depth: int,
        num_heads: int,
        embed_stride: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        # kernel=7 for stage 0 (large receptive field), kernel=3 for subsequent
        kernel = 7 if embed_stride == 4 else 3
        padding = kernel // 2
        self.embed = _ConvEmbed(
            in_ch, dim, kernel=kernel, stride=embed_stride, padding=padding
        )
        # Paper §3.2 / Table 1: K/V conv-projection uses stride=2 in *all*
        # three stages (Q is always stride=1).  This is the key trick that
        # makes CvT attention cheaper than ViT's O(N²).
        stride_kv = 2
        self.blocks = nn.ModuleList(
            [
                _CvTBlock(dim, num_heads, stride_kv, mlp_ratio, dropout)
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> tuple[Tensor, int, int]:  # type: ignore[override]
        # x: (B, C_in, H_in, W_in)
        x_spatial, H, W = cast(tuple[Tensor, int, int], self.embed(x))
        # Flatten to sequence: (B, H*W, C)
        B, C, _H, _W = x_spatial.shape
        tokens = x_spatial.reshape(B, C, H * W).permute(0, 2, 1)
        for blk in self.blocks:
            tokens = cast(Tensor, blk(tokens, H, W))  # type: ignore[arg-type]
        tokens = cast(Tensor, self.norm(tokens))
        # Reshape back to spatial for next stage
        x_out = tokens.permute(0, 2, 1).reshape(B, C, H, W)
        return x_out, H, W


# ---------------------------------------------------------------------------
# Body builder
# ---------------------------------------------------------------------------


def _build_stages(config: CvTConfig) -> tuple[list[_CvTStage], list[FeatureInfo]]:
    stages: list[_CvTStage] = []
    in_ch = config.in_channels
    cum_stride = 1
    fi: list[FeatureInfo] = []
    for i, (dim, depth, heads, stride) in enumerate(
        zip(config.dims, config.depths, config.num_heads, config.embed_strides)
    ):
        stages.append(
            _CvTStage(
                in_ch, dim, depth, heads, stride, config.mlp_ratio, config.dropout
            )
        )
        in_ch = dim
        cum_stride *= stride
        fi.append(FeatureInfo(stage=i + 1, num_channels=dim, reduction=cum_stride))
    return stages, fi


# ---------------------------------------------------------------------------
# CvT backbone
# ---------------------------------------------------------------------------


class CvT(PretrainedModel, BackboneMixin):
    r"""CvT backbone (Wu et al., 2021).

    The Convolutional vision Transformer (CvT) reinjects the locality
    inductive bias of CNNs into ViT through two changes: (i) every
    stage starts with an *overlapping convolutional token embedding*
    (kernel > stride) that produces a new lower-resolution token grid,
    and (ii) the Q / K / V projections inside every self-attention are
    *depthwise-separable convolutions* rather than linear maps:

    .. math::

        Q = \mathrm{Flatten}(\mathrm{DWConv}_q(x_{2D})),\quad
        K = \mathrm{Flatten}(\mathrm{DWConv}_k(x_{2D})),\quad
        V = \mathrm{Flatten}(\mathrm{DWConv}_v(x_{2D})),

    where :math:`x_{2D}` is the spatially reshaped token map.  When the
    K / V depthwise conv uses stride 2 the attention becomes *strided*,
    reducing :math:`N` by 4x inside attention without losing the
    full-resolution output for the next block.  CvT drops positional
    embeddings entirely — locality is supplied implicitly by the
    convolutional projections.

    :meth:`forward_features` returns the mean-pooled token feature
    :math:`(B, \texttt{dims[-1]})` over the final stage's tokens.

    Parameters
    ----------
    config : CvTConfig
        Frozen dataclass specifying ``dims``, ``depths``, ``num_heads``,
        ``embed_strides``, ``mlp_ratio``, ``dropout``, and
        ``in_channels``.  See :class:`CvTConfig`.

    Attributes
    ----------
    stages : nn.ModuleList
        Three :class:`_CvTStage` modules, each composed of a
        convolutional token embedding plus a stack of CvT blocks.
    feature_info : list[FeatureInfo]
        Three-stage feature description with cumulative reductions
        given by the cumulative product of ``config.embed_strides``.

    Notes
    -----
    Reference: Haiping Wu *et al.*, *"CvT: Introducing Convolutions to
    Vision Transformers"*, ICCV 2021,
    `arXiv:2103.15808 <https://arxiv.org/abs/2103.15808>`_.

    Examples
    --------
    Build a CvT-13 backbone and run a forward pass:

    >>> import lucid
    >>> from lucid.models.vision.cvt import CvT, CvTConfig
    >>> model = CvT(CvTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                       # (B, dims[-1])
    (1, 384)
    """

    config_class: ClassVar[type[CvTConfig]] = CvTConfig
    base_model_prefix: ClassVar[str] = "cvt"

    def __init__(self, config: CvTConfig) -> None:
        super().__init__(config)
        stage_list, fi = _build_stages(config)
        self.stages = nn.ModuleList(stage_list)  # type: ignore[arg-type]
        self._feature_info = fi

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        for stage in self.stages:
            x, _H, _W = cast(tuple[Tensor, int, int], stage(x))
        # x: (B, C, H, W) — flatten then mean
        return x.flatten(2).mean(dim=2)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# CvT for image classification
# ---------------------------------------------------------------------------


class CvTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""CvT with a linear classification head (Wu et al., 2021).

    Wraps the same three-stage trunk as :class:`CvT` and adds a global
    mean pool over tokens, a final LayerNorm, and a single
    :class:`nn.Linear` classification head:

    .. math::

        \text{logits} = W_{\text{head}}\,
            \mathrm{LN}\!\bigl(\mathrm{Mean}(z^{L})\bigr)
            + b_{\text{head}}.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy
    loss in the same pass.

    Parameters
    ----------
    config : CvTConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories.  See :class:`CvTConfig`.

    Attributes
    ----------
    stages : nn.ModuleList
        Three CvT stages.
    head_norm : nn.LayerNorm
        Final LayerNorm applied to the mean-pooled feature.
    classifier : nn.Linear
        Final linear projection of width ``(num_classes, dims[-1])``.

    Notes
    -----
    Reference: Wu *et al.*, *"CvT: Introducing Convolutions to Vision
    Transformers"*, ICCV 2021.  CvT-13 / CvT-21 reach **81.6% / 82.5%
    top-1 on ImageNet-1k** at 224x224 (Table 1 of the paper).

    Examples
    --------
    End-to-end inference with the default CvT-13 classifier:

    >>> import lucid
    >>> from lucid.models.vision.cvt import (
    ...     CvTConfig, CvTForImageClassification,
    ... )
    >>> model = CvTForImageClassification(CvTConfig())
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)
    """

    config_class: ClassVar[type[CvTConfig]] = CvTConfig
    base_model_prefix: ClassVar[str] = "cvt"

    def __init__(self, config: CvTConfig) -> None:
        super().__init__(config)
        stage_list, _ = _build_stages(config)
        self.stages = nn.ModuleList(stage_list)  # type: ignore[arg-type]
        self.head_norm = nn.LayerNorm(config.dims[-1])
        self._build_classifier(
            config.dims[-1], config.num_classes, dropout=config.dropout
        )

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        for stage in self.stages:
            x, _H, _W = cast(tuple[Tensor, int, int], stage(x))
        # x: (B, C, H, W) → (B, C) via global avg pool
        feat = x.flatten(2).mean(dim=2)
        feat = cast(Tensor, self.head_norm(feat))
        logits = cast(Tensor, self.classifier(feat))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
