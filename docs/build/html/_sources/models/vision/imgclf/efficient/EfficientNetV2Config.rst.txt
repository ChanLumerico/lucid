EfficientNetV2Config
====================

.. autoclass:: lucid.models.EfficientNetV2Config

`EfficientNetV2Config` stores the block sequence and classifier settings used by
:class:`lucid.models.EfficientNet_V2`. It defines the fused and MBConv layout,
classifier size, and dropout schedule for EfficientNet-v2 variants.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class EfficientNetV2Config:
        block_cfg: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | list[list[object]]
        num_classes: int = 1000
        dropout: float = 0.2
        drop_path_rate: float = 0.2

Parameters
----------

- **block_cfg**:
  Sequence of block specs in the form
  `(fused, out_channels, kernel_size, stride, expansion, repeats, se_scale)`.
- **num_classes** (*int*):
  Number of output classes.
- **dropout** (*float*):
  Dropout probability applied before the classifier.
- **drop_path_rate** (*float*):
  Progressive drop-path rate applied across the repeated blocks.

Validation
----------

- `block_cfg` must contain at least one seven-value block spec.
- Each block spec must use a boolean `fused` flag.
- `out_channels`, `kernel_size`, `stride`, `expansion`, and `repeats` must be positive integers.
- `se_scale` must be a non-negative integer.
- `num_classes` must be greater than 0.
- `dropout` must be in the range `[0, 1)`.
- `drop_path_rate` must be in the range `[0, 1]`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.EfficientNetV2Config(
        block_cfg=[
            (True, 24, 3, 1, 1, 1, 0),
            (False, 48, 3, 2, 4, 2, 4),
        ],
        num_classes=10,
        dropout=0.2,
        drop_path_rate=0.1,
    )
    model = models.EfficientNet_V2(config)
