XceptionConfig
==============

.. autoclass:: lucid.models.XceptionConfig

`XceptionConfig` stores the architectural settings used by
:class:`lucid.models.Xception`. It defines the classifier size, input channels,
and the channel widths and repeat counts used by the Xception entry, middle,
and exit flows.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class XceptionConfig:
        num_classes: int = 1000
        in_channels: int = 3
        stem_channels: tuple[int, int] | list[int] = (32, 64)
        entry_channels: tuple[int, int, int] | list[int] = (128, 256, 728)
        middle_channels: int = 728
        middle_repeats: int = 8
        exit_channels: tuple[int, int, int] | list[int] = (1024, 1536, 2048)

Parameters
----------

- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.
- **stem_channels** (*tuple[int, int] | list[int]*):
  Output widths of the two stem convolutions.
- **entry_channels** (*tuple[int, int, int] | list[int]*):
  Output widths of the three entry-flow residual blocks.
- **middle_channels** (*int*):
  Channel width used by the repeated middle-flow blocks.
- **middle_repeats** (*int*):
  Number of repeated middle-flow residual blocks.
- **exit_channels** (*tuple[int, int, int] | list[int]*):
  Channel widths used by the exit-flow block and the two final separable convolutions.

Validation
----------

- `num_classes`, `in_channels`, `middle_channels`, and `middle_repeats` must be greater than 0.
- `stem_channels` must contain exactly two positive integers.
- `entry_channels` and `exit_channels` must each contain exactly three positive integers.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.XceptionConfig(
        num_classes=10,
        in_channels=1,
        middle_repeats=8,
    )

    model = models.Xception(config)
