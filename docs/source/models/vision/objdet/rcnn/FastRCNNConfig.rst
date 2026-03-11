FastRCNNConfig
==============

.. autoclass:: lucid.models.FastRCNNConfig

`FastRCNNConfig` stores the shared backbone, RoI pooling shape, fully connected
head width, bounding-box normalization constants, dropout, and proposal
generator used by :class:`lucid.models.FastRCNN`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class FastRCNNConfig:
        backbone: nn.Module
        feat_channels: int
        num_classes: int
        pool_size: tuple[int, int] | list[int] = (7, 7)
        hidden_dim: int = 4096
        bbox_reg_means: tuple[float, ...] | list[float] = (0.0, 0.0, 0.0, 0.0)
        bbox_reg_stds: tuple[float, ...] | list[float] = (0.1, 0.1, 0.2, 0.2)
        dropout: float = 0.5
        proposal_generator: Callable[..., Tensor] | None = None

Parameters
----------

- **backbone** (*nn.Module*):
  Shared full-image feature extractor.
- **feat_channels** (*int*):
  Channel width of the backbone feature map.
- **num_classes** (*int*):
  Number of classification logits produced per RoI.
- **pool_size**:
  Output size used by RoIAlign.
- **hidden_dim** (*int*):
  Hidden width of the two-layer detection head.
- **bbox_reg_means**, **bbox_reg_stds**:
  Bounding-box regression normalization constants.
- **dropout** (*float*):
  Dropout probability used in the detection head.
- **proposal_generator**:
  Optional callable that returns pixel-space region proposals.

Validation
----------

- `backbone` must be an `nn.Module`.
- `feat_channels`, `num_classes`, and `hidden_dim` must be greater than `0`.
- `pool_size` must contain exactly two positive integers.
- `bbox_reg_means` must contain exactly four values.
- `bbox_reg_stds` must contain exactly four positive values.
- `dropout` must be in `[0, 1)`.
- `proposal_generator` must be callable or `None`.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
    )
    config = models.FastRCNNConfig(
        backbone=backbone,
        feat_channels=64,
        num_classes=4,
    )
    model = models.FastRCNN(config)
