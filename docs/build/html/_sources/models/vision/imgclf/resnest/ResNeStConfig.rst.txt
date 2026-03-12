ResNeStConfig
=============

.. autoclass:: lucid.models.ResNeStConfig

`ResNeStConfig` stores the split-attention backbone choices used by
:class:`lucid.models.ResNeSt`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class ResNeStConfig:
        layers: tuple[int, int, int, int] | list[int]
        base_width: int = 64
        stem_width: int = 32
        cardinality: int = 1
        radix: int = 2
        avd: bool = True
        num_classes: int = 1000
        in_channels: int = 3
        avg_down: bool = False
        channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
        block_args: dict[str, Any] = field(default_factory=dict)

Parameters
----------

- **layers** (*tuple[int, int, int, int] | list[int]*):
  Number of bottleneck blocks in each of the four stages.
- **base_width** (*int*):
  Width parameter used to derive grouped bottleneck channels.
- **stem_width** (*int*):
  Width of the deep stem.
- **cardinality** (*int*):
  Number of groups used by split-attention bottlenecks.
- **radix** (*int*):
  Number of attention splits per bottleneck. `0` falls back to the grouped-convolution path.
- **avd** (*bool*):
  Whether average downsampling is enabled inside the bottleneck.
- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.
- **avg_down** (*bool*):
  Whether projection shortcuts should downsample using average pooling before the 1x1 projection.
- **channels** (*tuple[int, int, int, int] | list[int]*):
  Output width of each residual stage.
- **block_args** (*dict[str, Any]*):
  Extra keyword arguments forwarded to the ResNeSt bottleneck, in addition to the
  preset split-attention settings.

Validation
----------

- `layers` and `channels` must each contain exactly four positive integers.
- `base_width`, `stem_width`, `cardinality`, `num_classes`, and `in_channels` must be greater than 0.
- `radix` must be greater than or equal to 0.
- `block_args` must be a dictionary.
