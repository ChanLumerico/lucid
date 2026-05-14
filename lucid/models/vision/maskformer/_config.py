"""MaskFormer configuration (Cheng et al., NeurIPS 2021)."""

from dataclasses import dataclass
from typing import ClassVar, Literal

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class MaskFormerConfig(ModelConfig):
    """Configuration for MaskFormer.

    MaskFormer (Cheng et al., NeurIPS 2021) reformulates semantic segmentation
    as mask classification: N learnable queries each predict a class label and
    a binary mask.  Predictions are matched to ground-truth segments via
    Hungarian matching during training.

    Architecture overview:
      Image → ResNet backbone → [C2, C3, C4, C5]
        → FPN Pixel Decoder → per-pixel embeddings (B, d_model, H/4, W/4)
        → Transformer Decoder (N queries attend to pixel embeddings)
        → Class head: Linear(d_model, num_classes+1) per query
        → Mask head: query-dot-pixel → binary mask per query

    Inference:
      seg_logits[b, k, h, w] = sum_n softmax(class_logits)[b,n,k]
                                      * sigmoid(mask_logits)[b,n,h,w]
      Upsampled to input resolution and argmaxed for final prediction.

    Args:
        num_classes:       Number of semantic classes (background = class 0).
        in_channels:       Input image channels.
        backbone_layers:   ResNet layer counts (default ResNet-50: 3,4,6,3).
        d_model:           Transformer embedding dimension.
        n_head:            Number of attention heads.
        num_encoder_layers: Pixel decoder transformer encoder depth.
        num_decoder_layers: Query transformer decoder depth.
        dim_feedforward:   FFN inner dimension in each transformer layer.
        dropout:           Dropout probability.
        num_queries:       Number of learnable object queries N.
        fpn_out_channels:  FPN lateral / output channel width.
    """

    model_type: ClassVar[str] = "maskformer"

    num_classes: int = 150
    in_channels: int = 3

    # Backbone
    backbone_layers: tuple[int, int, int, int] = (3, 4, 6, 3)  # ResNet-50
    backbone_block: Literal["basic", "bottleneck"] = "bottleneck"

    # Transformer
    d_model: int = 256
    n_head: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    num_queries: int = 100

    # FPN pixel decoder
    fpn_out_channels: int = 256

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_layers", tuple(self.backbone_layers))
