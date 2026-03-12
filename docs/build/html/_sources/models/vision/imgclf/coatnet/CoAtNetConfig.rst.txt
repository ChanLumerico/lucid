CoAtNetConfig
=============

.. autoclass:: lucid.models.CoAtNetConfig

`CoAtNetConfig` stores the stage layout and classifier settings used by
:class:`lucid.models.CoAtNet`. It defines the input resolution, stage depths,
stage widths, attention width, block pattern, and optional scaled tandem stage.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class CoAtNetConfig:
        img_size: tuple[int, int] | list[int]
        in_channels: int
        num_blocks: tuple[int, ...] | list[int]
        channels: tuple[int, ...] | list[int]
        num_classes: int = 1000
        num_heads: int = 32
        block_types: tuple[str, ...] | list[str] = ("C", "C", "T", "T")
        scaled_num_blocks: tuple[int, int] | list[int] | None = None
        scaled_channels: tuple[int, int] | list[int] | None = None

Parameters
----------

- **img_size**:
  Input image size as `(height, width)`.
- **in_channels** (*int*):
  Number of input channels.
- **num_blocks**:
  Five-stage block counts for the CoAtNet hierarchy.
- **channels**:
  Five-stage channel widths for the CoAtNet hierarchy.
- **num_classes** (*int*):
  Number of output classes.
- **num_heads** (*int*):
  Number of attention heads used by transformer stages.
- **block_types**:
  Four-stage block pattern after the convolutional stem, using `"C"` and `"T"`.
- **scaled_num_blocks**:
  Optional two-stage tandem depths used by the scaled CoAtNet variants.
- **scaled_channels**:
  Optional tandem stage widths paired with `scaled_num_blocks`.

Validation
----------

- `img_size` must contain exactly two integers greater than or equal to 32.
- `in_channels`, `num_classes`, and `num_heads` must be greater than 0.
- `num_blocks` and `channels` must contain exactly five positive integers.
- `block_types` must contain exactly four values drawn from `"C"` and `"T"`.
- `scaled_num_blocks` and `scaled_channels` must either both be omitted or both be two-value sequences of positive integers.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.CoAtNetConfig(
        img_size=(32, 32),
        in_channels=3,
        num_blocks=(1, 1, 1, 1, 1),
        channels=(8, 16, 32, 64, 128),
        num_classes=10,
        num_heads=4,
    )
    model = models.CoAtNet(config)
