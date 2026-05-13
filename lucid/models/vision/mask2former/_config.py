"""Mask2Former configuration (Cheng et al., CVPR 2022)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class Mask2FormerConfig(ModelConfig):
    """Configuration for Mask2Former.

    Mask2Former (Cheng et al., CVPR 2022) improves over MaskFormer with three
    key changes:

    1. **Masked cross-attention**: each decoder layer restricts query attention
       to within the predicted mask region from the previous layer.
    2. **Multi-scale features**: decoder layers alternate attending to
       different FPN feature levels (P3, P4, P5 cycling).
    3. **Improved pixel decoder**: FPN with multiple output scales.

    Architecture overview:
      Image → ResNet backbone → [C2, C3, C4, C5]
        → Multi-scale FPN Pixel Decoder → {P3, P4, P5} at fpn_out_channels
                                        + P2 projected to d_model
        → Transformer decoder (masked cross-attention cycling FPN levels)
        → Class head + mask head (same as MaskFormer)

    Args:
        num_classes:        Number of semantic classes.
        in_channels:        Input image channels.
        backbone_layers:    ResNet layer counts (default ResNet-50: 3,4,6,3).
        d_model:            Transformer embedding dimension.
        n_head:             Number of attention heads.
        num_encoder_layers: Pixel decoder encoder depth.
        num_decoder_layers: Query transformer decoder depth.
        dim_feedforward:    FFN inner dimension.
        dropout:            Dropout probability.
        num_queries:        Number of learnable object queries N.
        fpn_out_channels:   FPN lateral / output channel width.
        num_feature_levels: Number of multi-scale FPN levels used in decoder
                            cross-attention (cycles through them).
    """

    model_type: ClassVar[str] = "mask2former"

    num_classes: int = 150
    in_channels: int = 3

    # Backbone
    backbone_layers: tuple[int, int, int, int] = (3, 4, 6, 3)  # ResNet-50

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

    # Multi-scale decoder cross-attention
    num_feature_levels: int = 3  # P3, P4, P5

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_layers", tuple(self.backbone_layers))
