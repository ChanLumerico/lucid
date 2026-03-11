FasterRCNNConfig
================

.. autoclass:: lucid.models.FasterRCNNConfig

`FasterRCNNConfig` stores the shared backbone, feature-map channel width,
anchor generation settings, RoI pooling shape, and fully connected detection
head settings used by :class:`lucid.models.FasterRCNN`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class FasterRCNNConfig:
        backbone: nn.Module
        feat_channels: int
        num_classes: int
        use_fpn: bool = False
        anchor_sizes: tuple[int, ...] | list[int] = (128, 256, 512)
        aspect_ratios: tuple[float, ...] | list[float] = (0.5, 1.0, 2.0)
        anchor_stride: int = 16
        pool_size: tuple[int, int] | list[int] = (7, 7)
        hidden_dim: int = 4096
        dropout: float = 0.5

Parameters
----------

- **backbone** (*nn.Module*):
  Shared feature extractor used before the RPN and RoI head.
- **feat_channels** (*int*):
  Channel width of the feature map consumed by the RPN and detection head.
- **num_classes** (*int*):
  Number of classification logits produced per proposal.
- **use_fpn** (*bool*):
  Whether the backbone returns an FPN feature pyramid for multiscale RoIAlign.
- **anchor_sizes**, **aspect_ratios**:
  Anchor scales and aspect ratios used by the RPN anchor generator.
- **anchor_stride** (*int*):
  Spatial stride used when tiling anchors over the feature map.
- **pool_size**:
  Output size used by `ROIAlign` or `MultiScaleROIAlign`.
- **hidden_dim** (*int*):
  Hidden width of the two-layer detection head.
- **dropout** (*float*):
  Dropout probability used in the detection head.

Validation
----------

- `backbone` must be an `nn.Module`.
- `feat_channels`, `num_classes`, `anchor_stride`, and `hidden_dim` must be greater than `0`.
- `use_fpn` must be a `bool`.
- `anchor_sizes` and `aspect_ratios` must each contain at least one positive value.
- `pool_size` must contain exactly two positive integers.
- `dropout` must be in `[0, 1)`.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    config = models.FasterRCNNConfig(
        backbone=backbone,
        feat_channels=64,
        num_classes=4,
    )
    model = models.FasterRCNN(config)
