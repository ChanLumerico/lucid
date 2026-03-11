ConvNeXtConfig
==============

.. autoclass:: lucid.models.ConvNeXtConfig

`ConvNeXtConfig` stores the stage layout and classifier settings used by
:class:`lucid.models.ConvNeXt`. It defines the four-stage depth profile,
stage widths, classifier size, drop-path rate, and layer-scale initialization.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class ConvNeXtConfig:
        num_classes: int = 1000
        depths: tuple[int, int, int, int] | list[int] = (3, 3, 9, 3)
        dims: tuple[int, int, int, int] | list[int] = (96, 192, 384, 768)
        drop_path: float = 0.0
        layer_scale_init: float = 1e-6

Parameters
----------

- **num_classes** (*int*):
  Number of output classes.
- **depths**:
  Four-stage block counts for the ConvNeXt hierarchy.
- **dims**:
  Four-stage channel widths for the ConvNeXt hierarchy.
- **drop_path** (*float*):
  Global drop-path rate distributed across the stage blocks.
- **layer_scale_init** (*float*):
  Initial layer-scale value used inside ConvNeXt blocks.

Validation
----------

- `num_classes` must be greater than 0.
- `depths` and `dims` must contain exactly four positive integers.
- `drop_path` must be in the range `[0, 1]`.
- `layer_scale_init` must be greater than or equal to 0.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.ConvNeXtConfig(
        num_classes=10,
        depths=(2, 2, 6, 2),
        dims=(64, 128, 256, 512),
        drop_path=0.1,
        layer_scale_init=0.0,
    )
    model = models.ConvNeXt(config)
