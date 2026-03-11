VGGNetConfig
============

.. autoclass:: lucid.models.VGGNetConfig

`VGGNetConfig` stores the architectural choices used by :class:`lucid.models.VGGNet`.
It defines the convolution and pooling schedule, the output class count, the
input channel count, the classifier hidden widths, and the dropout probability
used in the classifier head.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class VGGNetConfig:
        conv_config: list[int | str]
        num_classes: int = 1000
        in_channels: int = 3
        dropout: float = 0.5
        classifier_hidden_features: tuple[int, int] = (4096, 4096)

Parameters
----------

- **conv_config** (*list[int | str]*): Convolution/pooling schedule where positive
  integers denote convolution output channels and `"M"` denotes max pooling.
- **num_classes** (*int*): Number of output classes.
- **in_channels** (*int*): Number of channels in the input image tensor.
- **dropout** (*float*): Dropout probability used in the classifier head.
- **classifier_hidden_features** (*tuple[int, int]*): Hidden widths of the two
  classifier linear layers.

Validation
----------

- `conv_config` must not be empty and must include at least 1 conv layer.
- Each `conv_config` entry must be a positive integer or `"M"`.
- `num_classes` and `in_channels` must be greater than 0.
- `dropout` must be in the range `[0.0, 1.0)`.
- `classifier_hidden_features` must contain exactly 2 positive integers.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.VGGNetConfig(
        conv_config=[64, "M", 128, "M", 256, 256, "M"],
        num_classes=10,
        in_channels=1,
        classifier_hidden_features=(512, 256),
        dropout=0.25,
    )

    model = models.VGGNet(config)
