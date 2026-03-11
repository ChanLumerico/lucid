EfficientDetConfig
==================

.. autoclass:: lucid.models.EfficientDetConfig

`EfficientDetConfig` stores the compound scaling coefficient, anchor count, and
class count used by :class:`lucid.models.EfficientDet`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class EfficientDetConfig:
        compound_coef: Literal[0, 1, 2, 3, 4, 5, 6, 7] = 0
        num_anchors: int = 9
        num_classes: int = 80

Parameters
----------

- **compound_coef** (*Literal[0-7]*):
  Compound scaling coefficient controlling the EfficientNet backbone width/depth
  and the BiFPN/head depth.
- **num_anchors** (*int*):
  Number of anchors per feature location. The current implementation requires `9`.
- **num_classes** (*int*):
  Number of target object classes predicted by the detector.

Validation
----------

- `compound_coef` must be an integer in `[0, 7]`.
- `num_anchors` must be `9` for the current anchor generator.
- `num_classes` must be greater than `0`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.EfficientDetConfig(compound_coef=0, num_classes=3)
    model = models.EfficientDet(config)
