MobileNetV3Config
=================

.. autoclass:: lucid.models.MobileNetV3Config

`MobileNetV3Config` stores the bottleneck sequence used by
:class:`lucid.models.MobileNet_V3`. It defines the MobileNet-v3 bottleneck
layout, stem width, classifier width, and classifier settings.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class MobileNetV3Config:
        bottleneck_cfg: tuple[tuple[object, ...], ...] | list[list[object]]
        last_channels: int
        num_classes: int = 1000
        in_channels: int = 3
        stem_channels: int = 16
        dropout: float = 0.2

Parameters
----------

- **bottleneck_cfg**:
  Sequence of bottleneck specs in the form
  `(kernel_size, mid_channels, out_channels, use_se, use_hswish, stride, se_reduction)`.
- **last_channels** (*int*):
  Output width of the penultimate classifier layer.
- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.
- **stem_channels** (*int*):
  Output width of the initial stem convolution.
- **dropout** (*float*):
  Dropout probability applied before the final classifier.

Validation
----------

- `bottleneck_cfg` must contain at least one seven-value bottleneck spec.
- Kernel, channel, stride, and reduction values inside each bottleneck spec must be positive integers.
- `use_se` and `use_hswish` must be booleans.
- `last_channels`, `num_classes`, `in_channels`, and `stem_channels` must be greater than 0.
- `dropout` must be in the range `[0, 1)`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.MobileNetV3Config(
        bottleneck_cfg=[
            (3, 16, 16, True, False, 2, 2),
            (3, 72, 24, False, False, 2, 4),
        ],
        last_channels=1024,
        num_classes=10,
        in_channels=1,
    )
    model = models.MobileNet_V3(config)
