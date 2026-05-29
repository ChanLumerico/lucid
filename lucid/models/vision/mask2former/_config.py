"""Mask2Former configuration (Cheng et al., CVPR 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig
from lucid.models._meta import model_family_meta


@model_family_meta(
    canonical_name="Mask2Former",
    citation=(
        'Cheng, Bowen, et al. "Masked-attention Mask Transformer for '
        'Universal Image Segmentation." Proceedings of the IEEE/CVF '
        "Conference on Computer Vision and Pattern Recognition, 2022, "
        "pp. 1290–1299."
    ),
    theory=r"""
    Mask2Former generalises MaskFormer into a single architecture
    that wins on **semantic**, **instance**, and **panoptic**
    segmentation simultaneously.  It keeps the mask-classification
    framing — :math:`N` object queries each predict a class and a
    binary mask — and improves it with three orthogonal changes.

    **Masked cross-attention.**  In every decoder layer the queries
    attend only to feature locations that the *previous* layer
    already considered foreground for that query:

    .. math::

        \mathrm{attn}_{ij} = \mathrm{softmax}_j\!\bigl(
            Q_i K_j^{\top} / \sqrt{d} + \mathcal{M}_{ij}^{\,\ell-1}
        \bigr),

    where :math:`\mathcal{M}^{\ell-1}_{ij} = 0` if the predicted
    binary mask of query :math:`i` at layer :math:`\ell - 1`
    activates pixel :math:`j` and :math:`-\infty` otherwise.  This
    confines query updates to plausible mask regions and accelerates
    convergence.

    **Multi-scale features.**  The decoder layers cycle through three
    pyramid levels :math:`(P_3, P_4, P_5)` of an enhanced FPN-style
    *pixel decoder*, so high-resolution semantic detail and broad
    contextual features are both available to every query.

    **Improved pixel decoder.**  A multi-scale deformable-attention
    encoder fuses the backbone feature maps before producing the
    per-pixel embeddings against which the query embeddings are dotted
    to form binary masks.

    Training uses the same Hungarian-matched class CE + mask BCE/Dice
    losses as MaskFormer (with auxiliary losses at intermediate
    decoder layers).  Mask2Former sets a new SOTA on each of ADE20K,
    Cityscapes, and COCO with a *single* model, eliminating the need
    for task-specific architectures.
    """,
)
@dataclass(frozen=True)
class Mask2FormerConfig(ModelConfig):
    """Configuration for Mask2Former (Cheng et al., CVPR 2022).

    The field set mirrors the reference framework's ``Mask2FormerConfig``
    so the pretrained-weight converter is a near-identity key map.  The
    pipeline is:

      Image → Swin backbone → [stage1..4] feature maps
        → MSDeformAttn pixel decoder → 3 multi-scale memory levels
                                     + 1/4-scale mask features
        → 9-layer masked-attention transformer decoder (cycling levels)
        → class head (Linear → K+1) + mask head (MLP → dot mask features)

    Args:
        num_classes:           Number of semantic classes (foreground; the
                               class head emits ``num_classes + 1``).
        in_channels:           Input image channels.
        swin_embed_dim:        Swin patch-embedding dimension.
        swin_depths:           Swin per-stage block counts.
        swin_num_heads:        Swin per-stage head counts.
        swin_window_size:      Swin attention window size.
        swin_mlp_ratio:        Swin MLP expansion ratio.
        d_model:               Transformer / pixel-decoder feature dim.
        mask_feature_size:     Per-pixel mask-feature channel width.
        n_head:                Number of attention heads.
        num_encoder_layers:    Deformable pixel-decoder encoder depth.
        encoder_feedforward_dim: Pixel-decoder FFN inner dim.
        num_decoder_layers:    Transformer decoder depth (the decoder uses
                               ``num_decoder_layers - 1`` masked layers; the
                               extra slot is the pre-layer mask prediction).
        dim_feedforward:       Transformer-decoder FFN inner dim.
        dropout:               Dropout probability (0 at inference).
        num_queries:           Number of learnable object queries N.
        num_feature_levels:    Number of multi-scale memory levels (3).
        feature_strides:       Backbone output strides.
        common_stride:         Finest pixel-decoder stride.
    """

    model_type: ClassVar[str] = "mask2former"

    num_classes: int = 150
    in_channels: int = 3

    # Swin backbone
    swin_embed_dim: int = 96
    swin_depths: tuple[int, int, int, int] = (2, 2, 6, 2)
    swin_num_heads: tuple[int, int, int, int] = (3, 6, 12, 24)
    swin_window_size: int = 7
    swin_mlp_ratio: float = 4.0

    # Transformer / pixel decoder
    d_model: int = 256
    mask_feature_size: int = 256
    n_head: int = 8
    num_encoder_layers: int = 6
    encoder_feedforward_dim: int = 1024
    num_decoder_layers: int = 10
    dim_feedforward: int = 2048
    dropout: float = 0.0
    num_queries: int = 100

    # Multi-scale memory levels
    num_feature_levels: int = 3
    feature_strides: tuple[int, int, int, int] = (4, 8, 16, 32)
    common_stride: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "swin_depths", tuple(self.swin_depths))
        object.__setattr__(self, "swin_num_heads", tuple(self.swin_num_heads))
        object.__setattr__(self, "feature_strides", tuple(self.feature_strides))
