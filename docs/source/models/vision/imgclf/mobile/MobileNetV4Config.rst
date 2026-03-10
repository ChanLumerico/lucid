MobileNetV4Config
=================

.. autoclass:: lucid.models.MobileNetV4Config

`MobileNetV4Config` stores the dictionary-based stage specification used by
:class:`lucid.models.MobileNet_V4`. It keeps the per-layer block specs together
with the classifier size.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class MobileNetV4Config:
        cfg: dict[str, dict[str, Any]]
        num_classes: int = 1000

Parameters
----------

- **cfg** (*dict[str, dict[str, Any]]*):
  Dictionary describing the `conv0` and `layer1`-`layer5` stage specifications.
- **num_classes** (*int*):
  Number of output classes.

Validation
----------

- `cfg` must be a dictionary containing `conv0` and `layer1`-`layer5`.
- Every stage spec must define `block_name`, `num_blocks`, and `block_specs`.
- `block_name` must be one of `convbn`, `uib`, or `fused_ib`.
- `num_blocks` must be a positive integer and match the length of `block_specs`.
- `num_classes` must be greater than 0.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.MobileNetV4Config(cfg=custom_cfg, num_classes=10)
    model = models.MobileNet_V4(config)
