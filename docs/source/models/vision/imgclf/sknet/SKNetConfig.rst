SKNetConfig
===========

.. autoclass:: lucid.models.SKNetConfig

`SKNetConfig` stores the Selective Kernel hyperparameters used by
:class:`lucid.models.SKNet`. It defines the residual stage depths, Selective
Kernel branch settings, classifier size, and the shared ResNet stem options.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class SKNetConfig:
        block: Literal["basic", "bottleneck"]
        layers: tuple[int, int, int, int] | list[int]
        kernel_sizes: tuple[int, ...] | list[int] = (3, 5)
        base_width: int = 64
        cardinality: int = 1
        num_classes: int = 1000
        in_channels: int = 3
        stem_width: int = 64
        stem_type: Literal["deep"] | None = None
        avg_down: bool = False
        channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
        block_args: dict[str, Any] = field(default_factory=dict)

Parameters
----------

- **block** (*Literal["basic", "bottleneck"]*):
  Selects the SK residual block variant.
- **layers** (*tuple[int, int, int, int] | list[int]*):
  Number of SK blocks in each of the four residual stages.
- **kernel_sizes** (*tuple[int, ...] | list[int]*):
  Kernel sizes used by the Selective Kernel branches inside each block.
- **base_width** (*int*):
  Width scaling factor used to derive intermediate block widths.
- **cardinality** (*int*):
  Group count used by the grouped Selective Kernel branches.
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
  Extra keyword arguments forwarded to each SK block, in addition to the preset
  `kernel_sizes`, `base_width`, and `cardinality`.

Validation
----------

- `block` must be `"basic"` or `"bottleneck"`.
- `layers` and `channels` must each contain exactly four positive integers.
- `kernel_sizes` must contain at least one positive integer.
- `base_width`, `cardinality`, `num_classes`, `in_channels`, and `stem_width` must be greater than 0.
- `stem_type` must be `None` or `"deep"`.
- `block_args` must be a dictionary.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.SKNetConfig(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        kernel_sizes=[3, 5, 7],
        cardinality=32,
        base_width=4,
        num_classes=10,
        avg_down=True,
    )

    model = models.SKNet(config)
