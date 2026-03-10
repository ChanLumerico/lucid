InceptionResNetConfig
=====================

.. autoclass:: lucid.models.InceptionResNetConfig

`InceptionResNetConfig` stores the architectural choices used by
:class:`lucid.models.InceptionResNet`. It defines which Inception-ResNet
variant to build and the common runtime options shared by the v1 and v2 families.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class InceptionResNetConfig:
        variant: Literal["v1", "v2"]
        num_classes: int = 1000
        in_channels: int = 3
        dropout_prob: float = 0.8

Parameters
----------

- **variant** (*Literal["v1", "v2"]*): Inception-ResNet family variant to build.
- **num_classes** (*int*): Number of output classes.
- **in_channels** (*int*): Number of channels in the input image tensor.
- **dropout_prob** (*float*): Dropout probability used before the classifier.

Validation
----------

- `variant` must be one of `"v1"` or `"v2"`.
- `num_classes` and `in_channels` must be greater than 0.
- `dropout_prob` must be in the range `[0.0, 1.0)`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.InceptionResNetConfig(
        variant="v2",
        num_classes=10,
        in_channels=1,
        dropout_prob=0.25,
    )

    model = models.InceptionResNet(config)
