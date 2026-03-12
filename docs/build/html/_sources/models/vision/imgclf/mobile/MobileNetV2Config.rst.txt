MobileNetV2Config
=================

.. autoclass:: lucid.models.MobileNetV2Config

`MobileNetV2Config` stores the inverted-residual stage settings used by
:class:`lucid.models.MobileNet_V2`. It defines the per-stage expansion ratios,
repeat counts, stem width, final projection width, and classifier settings.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class MobileNetV2Config:
        stage_configs: tuple[tuple[int, int, int, int, int], ...] | list[tuple[int, int, int, int, int]]
        num_classes: int = 1000
        in_channels: int = 3
        stem_channels: int = 32
        last_channels: int = 1280
        dropout: float = 0.2

Parameters
----------

- **stage_configs**:
  Sequence of stage specs in the form `(in_channels, out_channels, expansion, repeats, stride)`.
- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.
- **stem_channels** (*int*):
  Output width of the initial stem convolution.
- **last_channels** (*int*):
  Output width of the final 1x1 convolution.
- **dropout** (*float*):
  Dropout probability applied before the classifier.

Validation
----------

- `stage_configs` must contain at least one stage and each stage must contain five positive integers.
- `num_classes`, `in_channels`, `stem_channels`, and `last_channels` must be greater than 0.
- `dropout` must be in the range `[0, 1)`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.MobileNetV2Config(
        stage_configs=[
            (32, 16, 1, 1, 1),
            (16, 24, 6, 2, 2),
        ],
        num_classes=10,
        in_channels=1,
    )
    model = models.MobileNet_V2(config)
