ConvNeXtV2Config
================

.. autoclass:: lucid.models.ConvNeXtV2Config

`ConvNeXtV2Config` stores the stage layout and classifier settings used by
:class:`lucid.models.ConvNeXt_V2`. It defines the four-stage depth profile,
stage widths, classifier size, and drop-path rate for ConvNeXt-v2 variants.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class ConvNeXtV2Config:
        num_classes: int = 1000
        depths: tuple[int, int, int, int] | list[int] = (3, 3, 9, 3)
        dims: tuple[int, int, int, int] | list[int] = (96, 192, 384, 768)
        drop_path: float = 0.0

Parameters
----------

- **num_classes** (*int*):
  Number of output classes.
- **depths**:
  Four-stage block counts for the ConvNeXt-v2 hierarchy.
- **dims**:
  Four-stage channel widths for the ConvNeXt-v2 hierarchy.
- **drop_path** (*float*):
  Global drop-path rate distributed across the stage blocks.

Validation
----------

- `num_classes` must be greater than 0.
- `depths` and `dims` must contain exactly four positive integers.
- `drop_path` must be in the range `[0, 1]`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.ConvNeXtV2Config(
        num_classes=10,
        depths=(2, 2, 6, 2),
        dims=(40, 80, 160, 320),
        drop_path=0.1,
    )
    model = models.ConvNeXt_V2(config)
