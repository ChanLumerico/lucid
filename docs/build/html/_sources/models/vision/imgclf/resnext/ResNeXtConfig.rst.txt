ResNeXtConfig
=============

.. autoclass:: lucid.models.ResNeXtConfig

`ResNeXtConfig` stores the grouped-convolution settings used by
:class:`lucid.models.ResNeXt`. It defines the stage depths, cardinality,
base width, classifier size, and the shared ResNet stem options.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class ResNeXtConfig:
        layers: tuple[int, int, int, int] | list[int]
        cardinality: int
        base_width: int
        num_classes: int = 1000
        in_channels: int = 3
        stem_width: int = 64
        stem_type: Literal["deep"] | None = None
        avg_down: bool = False
        channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
        block_args: dict[str, Any] = field(default_factory=dict)

Parameters
----------

- **layers** (*tuple[int, int, int, int] | list[int]*):
  Number of bottleneck blocks in each of the four stages.
- **cardinality** (*int*):
  Number of groups used by the grouped bottleneck convolution.
- **base_width** (*int*):
  Width per group used to derive the bottleneck channel count.
- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.
- **stem_width** (*int*):
  Width parameter used by the deep stem variant.
- **stem_type** (*Literal["deep"] | None*):
  Stem style. `None` uses the classic single 7x7 stem, while `"deep"` uses a three-layer stem.
- **avg_down** (*bool*):
  Whether projection shortcuts should downsample using average pooling before the 1x1 projection.
- **channels** (*tuple[int, int, int, int] | list[int]*):
  Output width of each residual stage.
- **block_args** (*dict[str, Any]*):
  Extra keyword arguments forwarded to each grouped bottleneck block, in addition to
  the preset `cardinality` and `base_width`.

Validation
----------

- `layers` and `channels` must each contain exactly four positive integers.
- `cardinality`, `base_width`, `num_classes`, `in_channels`, and `stem_width` must be greater than 0.
- `stem_type` must be `None` or `"deep"`.
- `block_args` must be a dictionary.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.ResNeXtConfig(
        layers=[3, 4, 23, 3],
        cardinality=32,
        base_width=8,
        num_classes=10,
        avg_down=True,
    )

    model = models.ResNeXt(config)
