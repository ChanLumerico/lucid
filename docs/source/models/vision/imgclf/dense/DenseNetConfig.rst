DenseNetConfig
==============

.. autoclass:: lucid.models.DenseNetConfig

`DenseNetConfig` stores the architectural settings used by
:class:`lucid.models.DenseNet`. It defines the dense block depths, growth rate,
initial stem width, classifier size, and optional DenseNet-specific bottleneck
and transition-compression hyperparameters.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class DenseNetConfig:
        block_config: tuple[int, int, int, int] | list[int]
        growth_rate: int = 32
        num_init_features: int = 64
        num_classes: int = 1000
        in_channels: int = 3
        bottleneck: int = 4
        compression: float = 0.5

Parameters
----------

- **block_config** (*tuple[int, int, int, int] | list[int]*):
  Number of dense layers in each of the four dense blocks.
- **growth_rate** (*int*):
  Number of output channels added by each dense layer.
- **num_init_features** (*int*):
  Number of output channels produced by the initial convolution stem.
- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.
- **bottleneck** (*int*):
  Expansion factor used by the 1x1 bottleneck convolution inside each dense layer.
- **compression** (*float*):
  Transition-layer compression ratio applied after each dense block except the last.

Validation
----------

- `block_config` must contain exactly four positive integers.
- `growth_rate`, `num_init_features`, `num_classes`, `in_channels`, and `bottleneck` must be greater than 0.
- `compression` must be in the range `(0, 1]`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.DenseNetConfig(
        block_config=[6, 12, 32, 32],
        growth_rate=32,
        num_init_features=64,
        num_classes=10,
        in_channels=1,
        compression=0.5,
    )

    model = models.DenseNet(config)
