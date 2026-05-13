"""DETR configuration (Carion et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class DETRConfig(ModelConfig):
    """Configuration for DETR (DEtection TRansformer).

    DETR (Carion et al., ECCV 2020) reformulates object detection as a direct
    set prediction problem.  A CNN backbone extracts image features, a
    Transformer encoder-decoder processes them with N learned object queries,
    and FFN heads predict class labels + normalised (cx, cy, w, h) boxes.
    Training uses the Hungarian algorithm to match predictions to ground-truth
    objects (no hand-crafted anchors, no NMS at inference).

    Architecture overview:
      Image → ResNet-50 (C5 feature map) → 1×1 projection (d_model)
        → flatten + 2-D sinusoidal positional encoding
      → Transformer encoder (N_enc layers, d_model, n_head)
      → Transformer decoder (N_dec layers, N_queries object queries)
      → FFN: cls head (num_classes + 1) + box head (4, sigmoid)

    Args:
        num_classes:     Number of foreground classes (background = class 0).
        in_channels:     Input image channels.

        -- Backbone --
        backbone_layers: ResNet-50 layer counts (default 3,4,6,3).

        -- Transformer --
        d_model:       Transformer embedding dimension.
        n_head:        Number of self-/cross-attention heads.
        num_encoder_layers: Encoder depth.
        num_decoder_layers: Decoder depth.
        dim_feedforward:   FFN inner dimension in each transformer layer.
        dropout:           Dropout in transformer.
        num_queries:   Number of object queries (N).

        -- FFN heads --
        num_bbox_layers:   MLP depth for bounding-box head.
        bbox_hidden_dim:   Hidden width inside the bbox MLP.

        -- Inference --
        score_thresh:  Minimum predicted class probability (before NMS).
    """

    model_type: ClassVar[str] = "detr"

    num_classes: int = 80
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

    # FFN heads
    num_bbox_layers: int = 3
    bbox_hidden_dim: int = 256

    # Inference
    score_thresh: float = 0.7

    def __post_init__(self) -> None:
        object.__setattr__(self, "backbone_layers", tuple(self.backbone_layers))
