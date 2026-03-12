LeNetConfig
===========

.. autoclass:: lucid.models.LeNetConfig

`LeNetConfig` stores the architectural choices used by :class:`lucid.models.LeNet`.
It defines the two convolution stages, the classifier widths, the flattened
input size for the first linear layer, and the activation module class used
between layers.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class LeNetConfig:
        conv_layers: list[dict[str, int]]
        clf_layers: list[int]
        clf_in_features: int
        base_activation: Type[nn.Module] = nn.Tanh

Parameters
----------

- **conv_layers** (*list[dict[str, int]]*): Two convolution layer configs. Each
  entry must define `out_channels`.
- **clf_layers** (*list[int]*): Output widths of the classifier linear layers.
- **clf_in_features** (*int*): Flattened feature size consumed by the first
  classifier layer.
- **base_activation** (*Type[nn.Module]*): Activation module class inserted after
  each convolution and between classifier layers.

Validation
----------

- `conv_layers` must contain exactly 2 entries.
- Each convolution config must include `out_channels`.
- `clf_layers` must not be empty.
- `clf_in_features` must be greater than 0.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    config = models.LeNetConfig(
        conv_layers=[
            {"out_channels": 6},
            {"out_channels": 16},
        ],
        clf_layers=[120, 84, 10],
        clf_in_features=16 * 5 * 5,
        base_activation=nn.Tanh,
    )

    model = models.LeNet(config)
