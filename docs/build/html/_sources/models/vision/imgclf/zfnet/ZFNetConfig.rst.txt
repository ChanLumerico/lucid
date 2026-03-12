ZFNetConfig
===========

.. autoclass:: lucid.models.ZFNetConfig

`ZFNetConfig` stores the architectural choices used by :class:`lucid.models.ZFNet`.
It controls the output class count, the number of input channels, the dropout
rate used in the classifier, and the hidden widths of the two fully connected
classifier layers.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class ZFNetConfig:
        num_classes: int = 1000
        in_channels: int = 3
        dropout: float = 0.5
        classifier_hidden_features: tuple[int, int] = (4096, 4096)

Parameters
----------

- **num_classes** (*int*): Number of output classes.
- **in_channels** (*int*): Number of channels in the input image tensor.
- **dropout** (*float*): Dropout probability applied before the first two
  classifier linear layers.
- **classifier_hidden_features** (*tuple[int, int]*): Hidden widths of the two
  classifier linear layers.

Validation
----------

- `num_classes` must be greater than 0.
- `in_channels` must be greater than 0.
- `dropout` must be in the range `[0.0, 1.0)`.
- `classifier_hidden_features` must contain exactly 2 positive integers.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.ZFNetConfig(
        num_classes=10,
        in_channels=1,
        dropout=0.25,
        classifier_hidden_features=(512, 256),
    )

    model = models.ZFNet(config)
