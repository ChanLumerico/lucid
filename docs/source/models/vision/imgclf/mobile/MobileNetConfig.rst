MobileNetConfig
===============

.. autoclass:: lucid.models.MobileNetConfig

`MobileNetConfig` stores the MobileNet-v1 settings used by
:class:`lucid.models.MobileNet`. It captures the width multiplier together with
the classifier size and input channel count.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class MobileNetConfig:
        width_multiplier: float = 1.0
        num_classes: int = 1000
        in_channels: int = 3

Parameters
----------

- **width_multiplier** (*float*):
  Width scaling factor applied to each MobileNet-v1 channel dimension.
- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.

Validation
----------

- `width_multiplier`, `num_classes`, and `in_channels` must be greater than 0.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.MobileNetConfig(width_multiplier=0.75, num_classes=10, in_channels=1)
    model = models.MobileNet(config)
