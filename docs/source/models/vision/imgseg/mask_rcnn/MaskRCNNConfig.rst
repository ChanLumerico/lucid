MaskRCNNConfig
==============

.. autoclass:: lucid.models.MaskRCNNConfig

`MaskRCNNConfig` stores the backbone, RPN settings, detection head dimensions,
and mask head dimensions used by :class:`lucid.models.MaskRCNN`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class MaskRCNNConfig:
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
        mask_pool_size: tuple[int, int] | list[int] = (14, 14)
        mask_hidden_channels: int = 256
        mask_out_size: int = 28

Parameters
----------

- **backbone** (*nn.Module*):
  Shared feature extractor for proposal, box, and mask heads.
- **feat_channels** (*int*):
  Number of channels in the backbone feature maps consumed by the heads.
- **num_classes** (*int*):
  Number of foreground classes.
- **use_fpn** (*bool*):
  Whether the backbone returns FPN multi-scale features.
- **anchor_sizes** (*tuple[int, ...] | list[int]*):
  RPN anchor sizes.
- **aspect_ratios** (*tuple[float, ...] | list[float]*):
  RPN anchor aspect ratios.
- **anchor_stride** (*int*):
  Anchor stride on the feature map.
- **pool_size** (*tuple[int, int] | list[int]*):
  RoI pooling size for classification and box regression.
- **hidden_dim** (*int*):
  Hidden dimension of the MLP detection head.
- **dropout** (*float*):
  Dropout probability for the detection head.
- **mask_pool_size** (*tuple[int, int] | list[int]*):
  RoI pooling size for mask features.
- **mask_hidden_channels** (*int*):
  Hidden channels in the mask head.
- **mask_out_size** (*int*):
  Final target mask size. The current implementation requires
  `mask_out_size == 2 * mask_pool_size[0]`.

Validation
----------

- `backbone` must be an `nn.Module`.
- `feat_channels`, `num_classes`, `anchor_stride`, `hidden_dim`,
  `mask_hidden_channels`, and `mask_out_size` must be greater than `0`.
- `use_fpn` must be a boolean.
- `anchor_sizes` and `aspect_ratios` must be non-empty and positive.
- `pool_size` and `mask_pool_size` must contain exactly two positive integers.
- `mask_pool_size` must be square.
- `dropout` must be in `[0, 1)`.
- `mask_out_size` must equal `2 * mask_pool_size[0]`.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    class SimpleBackbone(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Conv2d(3, 64, kernel_size=3, padding=1)

        def forward(self, x):
            return self.net(x)

    config = models.MaskRCNNConfig(
        backbone=SimpleBackbone(),
        feat_channels=64,
        num_classes=5,
        mask_pool_size=(14, 14),
        mask_out_size=28,
    )
    model = models.MaskRCNN(config)
