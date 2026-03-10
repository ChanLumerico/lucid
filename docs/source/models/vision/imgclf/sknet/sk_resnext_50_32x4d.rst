sk_resnext_50_32x4d
===================

.. autofunction:: lucid.models.sk_resnext_50_32x4d

The `sk_resnext_50_32x4d` function constructs an SK-ResNeXt-50 (32x4d) model.
This preset uses `SKNetConfig(block="bottleneck", layers=[3, 4, 6, 3], cardinality=32, base_width=4)`.

**Total Parameters**: 29,274,760

Function Signature
------------------

.. code-block:: python

    @register_model
    def sk_resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> SKNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `SKNetConfig`, excluding the preset
  `block`, `layers`, `cardinality`, and `base_width` fields.

Returns
-------

- **SKNet**:
  An SK-ResNeXt-50 (32x4d) model instance with the preset
  `(block="bottleneck", layers=[3, 4, 6, 3], cardinality=32, base_width=4)`.
