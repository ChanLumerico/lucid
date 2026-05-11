"""CoAtNet configuration dataclass (Dai et al., 2021)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class CoAtNetConfig(ModelConfig):
    """Configuration for CoAtNet variants.

    ``variant`` selects the preset.  CoAtNet uses 4 stages after a 2-conv stem:

      Stage 1 / Stage 2 — MBConv (C-stages)
      Stage 3 / Stage 4 — Relative-attention Transformer (T-stages)

    Channel layout (96, 192, 384, 768) expands across the four stages;
    ``stem_width`` gives the initial conv channel count before S1.

    CoAtNet-0 reference: timm coatnet_0 ≈ 25.6 M params.
    """

    model_type: ClassVar[str] = "coatnet"

    num_classes: int = 1000
    in_channels: int = 3
    image_size: int = 224
    variant: str = "coatnet_0"

    # blocks_per_stage: (S1, S2, S3, S4) — 4 stages
    blocks_per_stage: tuple[int, ...] = (2, 3, 5, 2)

    # Channel dims for each of the 4 stages (S1 through S4)
    dims: tuple[int, ...] = (96, 192, 384, 768)

    # Stem output channels (before S1)
    stem_width: int = 64

    # Attention heads per T-stage (S3 head, S4 head); derived from dim/dim_head=32
    # S3 dim=384 → 384//32=12 heads; S4 dim=768 → 768//32=24 heads
    attn_heads: tuple[int, ...] = (12, 24)

    # MBConv expansion ratio (expand_output style: mid = out_ch * expand)
    mbconv_expand: int = 4

    # Optional pre-classifier hidden linear layer (timm head_hidden_size=768)
    head_hidden_size: int | None = 768

    dropout: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "blocks_per_stage", tuple(self.blocks_per_stage))
        object.__setattr__(self, "dims", tuple(self.dims))
        object.__setattr__(self, "attn_heads", tuple(self.attn_heads))
