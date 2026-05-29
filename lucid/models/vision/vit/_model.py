"""Vision Transformer (ViT) backbone and classifier (Dosovitskiy et al., 2020).

Paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

Architecture:
    PatchEmbed  : Conv2d(patch_size, stride=patch_size) → flatten → Linear
    [CLS] token : prepended learnable token
    PosEmbed    : learnable (1, num_patches+1, dim)
    Dropout
    N × ViTBlock:
        LayerNorm → _Attention(qkv + proj) → residual
        LayerNorm → MLP(dim → mlp_dim → dim, GELU) → residual
    LayerNorm
    Head (backbone: CLS token; classifier: CLS → FC)
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
from lucid.models.vision.vit._config import ViTConfig

# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class _PatchEmbed(nn.Module):
    r"""Split an image into non-overlapping patches and linearly project each.

    Implemented as a single ``Conv2d`` whose kernel size and stride are both
    equal to ``patch_size``.  This fuses the "unfold + linear project" pair
    into one convolution, producing an output of shape
    :math:`(B, D, H/p, W/p)` which is then flattened to
    :math:`(B, N, D)` where :math:`N = (H/p)(W/p)` is the number of patches.

    Parameters
    ----------
    in_channels : int
        Number of input image channels (typically 3 for RGB).
    patch_size : int
        Spatial side length of each non-overlapping patch.
    dim : int
        Output feature dimension :math:`D` of every patch embedding.

    Attributes
    ----------
    proj : nn.Conv2d
        The single projection convolution with ``kernel_size=stride=patch_size``.
    """

    def __init__(self, in_channels: int, patch_size: int, dim: int) -> None:
        super().__init__()
        # A Conv2d with kernel=stride=patch_size extracts patches in one op.
        self.proj = nn.Conv2d(in_channels, dim, patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        # (B, C, H, W) → (B, dim, H/p, W/p) → (B, num_patches, dim)
        x = cast(Tensor, self.proj(x))
        B, C, H, W = x.shape
        return x.reshape(B, C, H * W).permute(0, 2, 1)


# ---------------------------------------------------------------------------
# Self-attention with timm-compatible key naming
# ---------------------------------------------------------------------------


class _Attention(nn.Module):
    r"""Multi-head self-attention using a single fused QKV projection.

    Computes scaled dot-product attention over a sequence of tokens:

    .. math::

        \text{Attention}(Q, K, V)
            = \text{softmax}\!\left(\frac{Q K^{\top}}{\sqrt{d_k}}\right) V

    where :math:`Q, K, V \in \mathbb{R}^{B \times H \times N \times d_k}` and
    :math:`d_k = D / H` is the per-head dimension.  The three projections
    are fused into a single linear layer ``self.qkv`` producing an output
    of width ``3 * dim``, which is split along the last axis.  State-dict
    key naming (``attn.qkv.weight``, ``attn.qkv.bias``, ``attn.proj.weight``,
    ``attn.proj.bias``) matches the reference framework convention for ViT,
    making pretrained weights directly loadable.

    Parameters
    ----------
    dim : int
        Hidden width :math:`D` of the input / output features.  Must be
        divisible by ``num_heads``.
    num_heads : int
        Number of attention heads :math:`H`.
    attn_drop : float
        Dropout probability applied to the post-softmax attention weights.

    Attributes
    ----------
    qkv : nn.Linear
        Fused projection mapping ``dim`` -> ``3 * dim``.
    proj : nn.Linear
        Output projection mapping ``dim`` -> ``dim``.
    attn_drop : nn.Dropout
        Dropout on attention weights.
    """

    def __init__(self, dim: int, num_heads: int, attn_drop: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # Single fused projection for Q, K, V — matches timm naming exactly.
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        B, N, C = x.shape
        H, D = self.num_heads, self.head_dim

        # (B, N, 3*C) → (B, N, 3, H, D) → (3, B, H, N, D)
        qkv = cast(Tensor, self.qkv(x))
        qkv = qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4)
        # Each: (B, H, N, D)
        q: Tensor = qkv[0]
        k: Tensor = qkv[1]
        v: Tensor = qkv[2]

        # Scaled dot-product attention
        attn: Tensor = q @ k.permute(0, 1, 3, 2) / self.scale
        attn = F.softmax(attn, dim=-1)
        attn = cast(Tensor, self.attn_drop(attn))

        # (B, H, N, D) → (B, N, H*D) = (B, N, C)
        out: Tensor = attn @ v
        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        return cast(Tensor, self.proj(out))


# ---------------------------------------------------------------------------
# MLP inside each ViT block
# ---------------------------------------------------------------------------


class _MLP(nn.Module):
    r"""Two-layer feed-forward MLP used inside every ViT block.

    Applies the position-wise feed-forward transformation

    .. math::

        \text{MLP}(x) = \text{Dropout}\!\big(W_2\,
        \text{Dropout}(\text{GELU}(W_1 x + b_1)) + b_2\big)

    where :math:`W_1 \in \mathbb{R}^{D' \times D}` and
    :math:`W_2 \in \mathbb{R}^{D \times D'}` with hidden width
    :math:`D' = \text{int}(D \cdot \text{mlp\_ratio})`.  The dropout layer
    is shared across both positions, following the original recipe.

    Parameters
    ----------
    dim : int
        Input / output dimensionality :math:`D`.
    mlp_dim : int
        Hidden dimensionality :math:`D'`, typically ``4 * dim``.
    dropout : float
        Dropout probability applied after each linear layer.
    """

    def __init__(self, dim: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = cast(Tensor, self.drop(F.gelu(cast(Tensor, self.fc1(x)))))
        return cast(Tensor, self.drop(cast(Tensor, self.fc2(x))))


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------


class _ViTBlock(nn.Module):
    r"""Single transformer encoder block in pre-norm configuration.

    Each block applies LayerNorm before attention and before the MLP, with
    residual connections wrapping both sub-layers:

    .. math::

        \begin{aligned}
        x &\leftarrow x + \text{Attention}(\text{LN}(x)) \\
        x &\leftarrow x + \text{MLP}(\text{LN}(x))
        \end{aligned}

    Pre-norm (LayerNorm *before* each sub-layer rather than after) is the
    convention used in the ViT paper because it improves optimization
    stability for deep stacks.

    Parameters
    ----------
    dim : int
        Hidden width :math:`D` shared by the input, output, and residual
        stream.
    num_heads : int
        Number of attention heads in the self-attention sub-layer.
    mlp_dim : int
        Hidden width of the MLP sub-layer (already-resolved
        ``int(dim * mlp_ratio)``).
    dropout : float
        Dropout probability inside the MLP feed-forward layer.
    attention_dropout : float
        Dropout probability on post-softmax attention weights.

    Attributes
    ----------
    norm1 : nn.LayerNorm
        LayerNorm applied before the attention sub-layer.
    attn : _Attention
        Multi-head self-attention sub-layer.
    norm2 : nn.LayerNorm
        LayerNorm applied before the MLP sub-layer.
    mlp : _MLP
        Position-wise feed-forward sub-layer.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        eps: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = _Attention(dim, num_heads, attn_drop=attention_dropout)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = _MLP(dim, mlp_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
        x = x + cast(Tensor, self.attn(cast(Tensor, self.norm1(x))))
        x = x + cast(Tensor, self.mlp(cast(Tensor, self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Shared trunk builder
# ---------------------------------------------------------------------------


def _build_trunk(cfg: ViTConfig) -> tuple[
    _PatchEmbed,
    Tensor,  # cls_token parameter
    Tensor,  # pos_embed parameter
    nn.Dropout,
    nn.ModuleList,  # blocks
    nn.LayerNorm,  # norm
    int,  # num_patches
]:
    num_patches = (cfg.image_size // cfg.patch_size) ** 2
    mlp_dim = int(cfg.dim * cfg.mlp_ratio)

    patch_embed = _PatchEmbed(cfg.in_channels, cfg.patch_size, cfg.dim)

    cls_token = nn.Parameter(lucid.zeros(1, 1, cfg.dim))
    pos_embed = nn.Parameter(lucid.zeros(1, num_patches + 1, cfg.dim))
    # Simple uniform initialisation; no sinusoidal needed for learned pos embed
    nn.init.trunc_normal_(pos_embed, std=0.02)
    nn.init.trunc_normal_(cls_token, std=0.02)

    drop = nn.Dropout(p=cfg.dropout)
    blocks = nn.ModuleList(
        [
            _ViTBlock(
                cfg.dim,
                cfg.num_heads,
                mlp_dim,
                cfg.dropout,
                cfg.attention_dropout,
                cfg.layer_norm_eps,
            )
            for _ in range(cfg.depth)
        ]
    )
    norm = nn.LayerNorm(cfg.dim, eps=cfg.layer_norm_eps)

    return patch_embed, cls_token, pos_embed, drop, blocks, norm, num_patches


# ---------------------------------------------------------------------------
# ViT backbone  (task="base")
# ---------------------------------------------------------------------------


class ViT(PretrainedModel, BackboneMixin):
    r"""Vision Transformer backbone (Dosovitskiy et al., 2020).

    A ViT first splits the input image into a sequence of non-overlapping
    patches and linearly projects each patch into a :math:`D`-dimensional
    embedding.  A learnable ``[CLS]`` token is prepended, a learnable
    positional embedding is added, and the resulting sequence is processed
    by ``depth`` stacked pre-norm transformer encoder blocks.  The final
    representation returned by :meth:`forward_features` is the CLS token
    embedding after a final LayerNorm — the standard feature used for
    classification or transfer learning.

    Concretely, the input sequence fed to the encoder is

    .. math::

        z_0 = [\,x_{\text{class}};\, x_p^1 E;\, x_p^2 E;\, \dots;\,
                  x_p^N E\,] + E_{\text{pos}},

    where :math:`x_p^i \in \mathbb{R}^{P^2 C}` is the :math:`i`-th flattened
    patch, :math:`E \in \mathbb{R}^{P^2 C \times D}` is the patch projection,
    :math:`E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}` is the learnable
    positional embedding, and :math:`N = (\text{image\_size} /
    \text{patch\_size})^2`.

    Use this backbone when you want features rather than logits — e.g.
    fine-tuning on a custom downstream head, contrastive pretraining, or
    feature extraction for retrieval.  For end-to-end image classification,
    use :class:`ViTForImageClassification` instead.

    Parameters
    ----------
    config : ViTConfig
        Frozen dataclass specifying ``image_size``, ``patch_size``, ``dim``,
        ``depth``, ``num_heads``, ``mlp_ratio``, ``dropout``,
        ``attention_dropout``, and ``in_channels``.  See :class:`ViTConfig`.

    Attributes
    ----------
    patch_embed : _PatchEmbed
        Patch-extraction convolution producing the initial token sequence.
    cls_token : Tensor
        Learnable ``[CLS]`` token, shape ``(1, 1, dim)``.  Initialized with
        truncated normal :math:`\mathcal{N}(0, 0.02^2)`.
    pos_embed : Tensor
        Learnable positional embedding, shape ``(1, num_patches + 1, dim)``.
        Initialized with truncated normal :math:`\mathcal{N}(0, 0.02^2)`.
    pos_drop : nn.Dropout
        Dropout applied right after adding positional embeddings.
    blocks : nn.ModuleList
        ``config.depth`` stacked :class:`_ViTBlock` encoder layers.
    norm : nn.LayerNorm
        Final LayerNorm applied before extracting the CLS token.
    feature_info : list[FeatureInfo]
        Single-stage feature description with channel count equal to
        ``config.dim`` and spatial reduction equal to ``config.patch_size``.

    Notes
    -----
    The classification head is **not** included in this class — only the
    backbone trunk.  :meth:`forward_features` returns a flat
    :math:`(B, D)` feature tensor, whereas :meth:`forward` wraps it in a
    :class:`BaseModelOutput` whose ``last_hidden_state`` has shape
    :math:`(B, 1, D)` for spatial consistency with other vision models.

    Reference: Alexey Dosovitskiy *et al.*, *"An Image is Worth 16x16
    Words: Transformers for Image Recognition at Scale"*, ICLR 2021,
    `arXiv:2010.11929 <https://arxiv.org/abs/2010.11929>`_.

    Examples
    --------
    Build a ViT-Base/16 backbone and run a single forward pass:

    >>> import lucid
    >>> from lucid.models.vision.vit import ViT, ViTConfig
    >>> cfg = ViTConfig()              # ViT-Base/16 defaults
    >>> model = ViT(cfg)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> feat = model.forward_features(x)
    >>> feat.shape                     # (B, dim)
    (1, 768)
    >>> out = model(x)
    >>> out.last_hidden_state.shape    # (B, 1, dim)
    (1, 1, 768)
    """

    config_class: ClassVar[type[ViTConfig]] = ViTConfig
    base_model_prefix: ClassVar[str] = "vit"

    cls_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        pe, ct, pos, drop, blocks, norm, num_patches = _build_trunk(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.pos_embed = pos
        self.pos_drop = drop
        self.blocks = blocks
        self.norm = norm
        self._num_patches = num_patches
        self._feature_info = [
            FeatureInfo(stage=1, num_channels=config.dim, reduction=config.patch_size),
        ]

    @property
    def feature_info(self) -> list[FeatureInfo]:
        return self._feature_info

    def forward_features(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        x = cast(Tensor, self.patch_embed(x))  # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, dim)
        x = lucid.cat([cls, x], dim=1)  # (B, N+1, dim)
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))
        return x[:, 0]  # CLS token: (B, dim)

    def forward(self, x: Tensor) -> BaseModelOutput:  # type: ignore[override]
        feat = self.forward_features(x)
        # Unsqueeze to (B, 1, dim) so BaseModelOutput is spatially consistent
        return BaseModelOutput(last_hidden_state=feat.unsqueeze(1))


# ---------------------------------------------------------------------------
# ViT for image classification  (task="image-classification")
# ---------------------------------------------------------------------------


class ViTForImageClassification(PretrainedModel, ClassificationHeadMixin):
    r"""Vision Transformer with a linear classification head on the CLS token.

    Wraps the same trunk as :class:`ViT` (patch embedding -> positional
    embedding -> ``depth`` transformer blocks -> final LayerNorm) and adds
    a single :class:`nn.Linear` classification head that maps the final
    CLS-token feature of dimension :math:`D` to ``config.num_classes``
    logits:

    .. math::

        \text{logits} = W_{\text{head}}\, z_L^{[0]} + b_{\text{head}},
        \qquad W_{\text{head}} \in \mathbb{R}^{C \times D}

    where :math:`z_L^{[0]}` is the post-LayerNorm CLS token at the final
    layer and :math:`C` is the number of classes.  This is the standard
    ImageNet recipe described in the original ViT paper.

    Pass ``labels`` to :meth:`forward` to compute the cross-entropy loss
    in the same pass — handy for HuggingFace-style training loops.

    Parameters
    ----------
    config : ViTConfig
        Architecture specification.  Must set ``num_classes`` to the
        desired number of output categories (default 1000 for ImageNet-1k).
        See :class:`ViTConfig`.

    Attributes
    ----------
    patch_embed : _PatchEmbed
        Patch-extraction convolution producing the initial token sequence.
    cls_token : Tensor
        Learnable CLS token, shape ``(1, 1, dim)``.
    pos_embed : Tensor
        Learnable positional embedding, shape ``(1, num_patches + 1, dim)``.
    pos_drop : nn.Dropout
        Dropout applied right after adding positional embeddings.
    blocks : nn.ModuleList
        Stacked transformer encoder blocks.
    norm : nn.LayerNorm
        Final LayerNorm applied before extracting the CLS token.
    head : nn.Linear
        Classification head with shape ``(num_classes, dim)``.  The
        attribute is intentionally named ``head`` so pretrained
        state-dicts using the standard ``head.weight`` / ``head.bias`` key
        names load directly.

    Notes
    -----
    Reference: Alexey Dosovitskiy *et al.*, *"An Image is Worth 16x16
    Words: Transformers for Image Recognition at Scale"*, ICLR 2021,
    `arXiv:2010.11929 <https://arxiv.org/abs/2010.11929>`_.

    Examples
    --------
    End-to-end inference with the default ViT-Base/16 classifier:

    >>> import lucid
    >>> from lucid.models.vision.vit import (
    ...     ViTConfig, ViTForImageClassification
    ... )
    >>> cfg = ViTConfig(num_classes=1000)
    >>> model = ViTForImageClassification(cfg)
    >>> x = lucid.randn(1, 3, 224, 224)
    >>> out = model(x)
    >>> out.logits.shape
    (1, 1000)

    Pass labels for joint loss computation during training:

    >>> labels = lucid.tensor([0])
    >>> out = model(x, labels=labels)
    >>> out.loss.shape
    ()
    """

    config_class: ClassVar[type[ViTConfig]] = ViTConfig
    base_model_prefix: ClassVar[str] = "vit"

    cls_token: Tensor
    pos_embed: Tensor

    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        pe, ct, pos, drop, blocks, norm, num_patches = _build_trunk(config)
        self.patch_embed = pe
        self.cls_token = ct
        self.pos_embed = pos
        self.pos_drop = drop
        self.blocks = blocks
        self.norm = norm
        # Named ``head`` to match timm's vit_base_patch16_224 state-dict keys.
        self.head = nn.Linear(config.dim, config.num_classes)

    def forward(  # type: ignore[override]
        self,
        x: Tensor,
        labels: Tensor | None = None,
    ) -> ImageClassificationOutput:
        B = x.shape[0]
        x = cast(Tensor, self.patch_embed(x))
        cls = self.cls_token.expand(B, -1, -1)
        x = lucid.cat([cls, x], dim=1)
        x = cast(Tensor, self.pos_drop(x + self.pos_embed))
        for blk in self.blocks:
            x = cast(Tensor, blk(x))
        x = cast(Tensor, self.norm(x))
        logits = cast(Tensor, self.head(x[:, 0]))

        loss: Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return ImageClassificationOutput(logits=logits, loss=loss)
