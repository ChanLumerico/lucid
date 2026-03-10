InceptionConfig
===============

.. autoclass:: lucid.models.InceptionConfig

`InceptionConfig` stores the architectural choices used by :class:`lucid.models.Inception`.
It defines which Inception variant to build and the common runtime options shared
by the v1, v3, and v4 families.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class InceptionConfig:
        variant: Literal["v1", "v3", "v4"]
        num_classes: int = 1000
        in_channels: int = 3
        use_aux: bool | None = None
        dropout_prob: float | None = None

Parameters
----------

- **variant** (*Literal["v1", "v3", "v4"]*): Inception family variant to build.
- **num_classes** (*int*): Number of output classes.
- **in_channels** (*int*): Number of channels in the input image tensor.
- **use_aux** (*bool | None*): Auxiliary classifier flag for v1/v3. For v4 it is
  normalized to `False`.
- **dropout_prob** (*float | None*): Optional dropout override. If omitted, each
  variant uses its existing default.

Validation
----------

- `variant` must be one of `"v1"`, `"v3"`, or `"v4"`.
- `num_classes` and `in_channels` must be greater than 0.
- `dropout_prob`, when provided, must be in the range `[0.0, 1.0)`.
- `use_aux` must be boolean for v1 and v3.
- Inception v4 does not support auxiliary classifiers.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.InceptionConfig(
        variant="v4",
        num_classes=10,
        in_channels=1,
        dropout_prob=0.25,
    )

    model = models.Inception(config)
