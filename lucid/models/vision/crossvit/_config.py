"""CrossViT configuration dataclass (Chen et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="CrossViT",
    citation=(
        'Chen, Chun-Fu (Richard), et al. "CrossViT: Cross-Attention '
        'Multi-Scale Vision Transformer for Image Classification." '
        "Proceedings of the IEEE/CVF International Conference on "
        "Computer Vision, 2021, pp. 357-366."
    ),
    theory=r"""
    CrossViT explicitly models *multi-scale* visual information by
    running two parallel ViT branches with different patch sizes and
    fusing them through a lightweight cross-attention mechanism.  A
    *small-patch* branch (e.g. patch size 12) processes a long sequence
    of fine-grained tokens, while a *large-patch* branch (e.g. patch
    size 16) processes a short sequence of coarse tokens.  Each branch
    has its own class token :math:`x_{\text{cls}}^{s}` /
    :math:`x_{\text{cls}}^{l}` and is independently transformer-encoded
    for :math:`L` blocks.

    Information is exchanged between branches only through the class
    tokens, using *cross-attention*: the class token of one branch
    attends to the patch tokens of the *other* branch as queries,
    leaving the other branch's patch tokens unchanged.  Concretely, for
    the small-to-large direction,

    .. math::

        \tilde{x}_{\text{cls}}^{\,l} = x_{\text{cls}}^{l} +
            \mathrm{Attn}\!\bigl(W_q x_{\text{cls}}^{l},
            W_k x_{\text{patch}}^{s}, W_v x_{\text{patch}}^{s}\bigr),

    and symmetrically for large-to-small.  Because cross-attention is
    only :math:`\mathcal{O}(N_s + N_l)` per direction, multi-scale
    fusion is essentially free relative to the per-branch self-attention
    cost.  The final classifier averages the two updated class tokens.
    The variants CrossViT-9 / 15 / 18 trade depth and width along this
    two-branch skeleton.
    """,
)
@dataclass(frozen=True)
class CrossViTConfig(ModelConfig):
    r"""Configuration dataclass for every CrossViT variant.

    ``CrossViTConfig`` is an immutable container that fully specifies
    the architecture of a CrossViT (Chen et al., 2021).  CrossViT
    couples two ViT branches that operate on *different patch sizes*
    (``small_patch`` and ``large_patch``) and exchanges information
    only through their CLS tokens via lightweight cross-attention.

    Parameters
    ----------
    num_classes : int, optional
        Number of output classes for the classification head.
        Defaults to ``1000`` (ImageNet-1k).
    in_channels : int, optional
        Number of input image channels.  Defaults to ``3`` (RGB).
    image_size : int, optional
        Spatial side length of the square input image.  Both
        ``small_patch`` and ``large_patch`` must divide ``image_size``.
        Defaults to ``224``.
    small_patch : int, optional
        Patch size for the *small-patch* branch (fine-grained tokens).
        Defaults to ``12``.
    large_patch : int, optional
        Patch size for the *large-patch* branch (coarse tokens).
        Defaults to ``16``.
    small_dim : int, optional
        Hidden width of the small-patch branch.  Defaults to ``192``.
    large_dim : int, optional
        Hidden width of the large-patch branch.  Defaults to ``384``.
    small_heads : int, optional
        Number of attention heads in the small-patch branch.  Defaults
        to ``3``.
    large_heads : int, optional
        Number of attention heads in the large-patch branch.  Defaults
        to ``6``.
    depth : int, optional
        Number of CrossViT stages.  Each stage applies a self-attention
        block on both branches followed by a bidirectional CLS-token
        cross-attention exchange.  Defaults to ``4``.
    mlp_ratio : float, optional
        MLP expansion ratio inside both branches.  Defaults to ``4.0``.
    dropout : float, optional
        Dropout probability after positional embeddings and inside
        the MLP blocks.  Defaults to ``0.0``.

    Attributes
    ----------
    model_type : ClassVar[str]
        Constant string ``"crossvit"`` used by the model registry.

    Notes
    -----
    Reference: Chun-Fu (Richard) Chen *et al.*, *"CrossViT:
    Cross-Attention Multi-Scale Vision Transformer for Image
    Classification"*, ICCV 2021,
    `arXiv:2103.14899 <https://arxiv.org/abs/2103.14899>`_.

    Examples
    --------
    Build a CrossViT-9 configuration with a 10-class head for CIFAR-10:

    >>> from lucid.models.vision.crossvit import CrossViTConfig
    >>> cfg = CrossViTConfig(
    ...     depth=3, small_dim=128, large_dim=256,
    ...     small_heads=4, large_heads=4, mlp_ratio=3.0,
    ...     num_classes=10,
    ... )
    >>> cfg.small_dim, cfg.large_dim, cfg.depth, cfg.num_classes
    (128, 256, 3, 10)
    """

    model_type: ClassVar[str] = "crossvit"

    num_classes: int = 1000
    in_channels: int = 3
    image_size: int = 224
    small_patch: int = 12
    large_patch: int = 16
    small_dim: int = 192
    large_dim: int = 384
    small_heads: int = 3
    large_heads: int = 6
    depth: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
