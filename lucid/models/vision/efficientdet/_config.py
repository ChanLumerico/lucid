"""EfficientDet configuration (Tan et al., 2020)."""

from dataclasses import dataclass
from typing import ClassVar

from lucid.models._base import ModelConfig


@dataclass(frozen=True)
class EfficientDetConfig(ModelConfig):
    """Configuration for EfficientDet.

    EfficientDet (Tan et al., CVPR 2020) applies compound scaling to object
    detection.  Each compound coefficient φ jointly scales the backbone
    (EfficientNet-Bφ), the BiFPN width/depth, and the prediction head
    depth/resolution.

    Architecture overview:
      Image → EfficientNet backbone (P3–P7 feature maps)
        → BiFPN (D_bifpn repeats, W_bifpn channels, fast normalised fusion)
      BiFPN outputs (P3–P7) → Class prediction head (shared conv)
                             → Box prediction head  (shared conv)

    BiFPN fast-normalised fusion:
      weight_i / (ε + Σ weight_j) — learnable positive weights (ε=1e-4)

    Args:
        num_classes:    Foreground classes (background not counted separately
                        — EfficientDet uses sigmoid + focal loss).
        in_channels:    Input image channels.

        -- Compound scaling (φ = 0–7) --
        phi:          Compound coefficient index.  Default 0 (EfficientDet-D0).

        -- Backbone (EfficientNet-B0–B7 widths/depths) --
        backbone_width_coeff:  Width multiplier (EfficientNet channel scaling).
        backbone_depth_coeff:  Depth multiplier (EfficientNet block repeat).
        backbone_drop_rate:    Stochastic depth / dropout.

        -- BiFPN --
        fpn_channels:   BiFPN channel width (W_bifpn).
        fpn_repeats:    Number of BiFPN stacking repetitions (D_bifpn).

        -- Prediction heads --
        head_repeats:   Depth of class/box conv heads (D_head).

        -- Anchors --
        anchor_scales:    Per-level anchor size multipliers.
        anchor_ratios:    Anchor aspect ratios.
        anchor_num_scales: Octave subdivisions per scale.

        -- Inference --
        score_thresh:   Minimum sigmoid class score.
        nms_thresh:     Per-class NMS threshold.
        max_detections: Maximum detections per image.
    """

    model_type: ClassVar[str] = "efficientdet"

    num_classes: int = 80
    in_channels: int = 3

    # Compound coefficient
    phi: int = 0

    # Backbone feature channel counts at P3/P4/P5 (extracted from EfficientNet)
    backbone_in_channels: tuple[int, int, int] = (40, 112, 320)

    # BiFPN
    fpn_channels: int = 64
    fpn_repeats: int = 3

    # Prediction head depth
    head_repeats: int = 3

    # Anchors: 3 scales × 3 ratios = 9 per cell
    anchor_scales: tuple[float, ...] = (1.0, 2.0 ** (1.0 / 3.0), 2.0 ** (2.0 / 3.0))
    anchor_ratios: tuple[float, ...] = (0.5, 1.0, 2.0)
    anchor_base_sizes: tuple[int, ...] = (32, 64, 128, 256, 512)

    # Inference
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    max_detections: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backbone_in_channels", tuple(self.backbone_in_channels)
        )
        object.__setattr__(self, "anchor_scales", tuple(self.anchor_scales))
        object.__setattr__(self, "anchor_ratios", tuple(self.anchor_ratios))
        object.__setattr__(self, "anchor_base_sizes", tuple(self.anchor_base_sizes))


# ---------------------------------------------------------------------------
# Compound-scaling lookup table (paper Table 1)
# ---------------------------------------------------------------------------

# Each entry: (phi, fpn_channels, fpn_repeats, head_repeats,
#              backbone_in_channels-P3/P4/P5, image_size)
_COMPOUND_PARAMS: dict[int, tuple[int, int, int, tuple[int, int, int]]] = {
    0: (64, 3, 3, (40, 112, 320)),
    1: (88, 4, 3, (40, 112, 320)),
    2: (112, 5, 3, (48, 120, 352)),
    3: (160, 6, 4, (48, 136, 384)),
    4: (224, 7, 4, (56, 160, 448)),
    5: (288, 7, 4, (64, 176, 512)),
    6: (384, 8, 5, (72, 200, 576)),
    7: (384, 8, 5, (80, 224, 640)),
}


def efficientdet_config(phi: int = 0, num_classes: int = 80) -> EfficientDetConfig:
    """Build the standard EfficientDet-D{phi} config from the compound table."""
    fpn_ch, fpn_rep, head_rep, bb_ch = _COMPOUND_PARAMS[phi]
    return EfficientDetConfig(
        num_classes=num_classes,
        phi=phi,
        backbone_in_channels=bb_ch,
        fpn_channels=fpn_ch,
        fpn_repeats=fpn_rep,
        head_repeats=head_rep,
    )
