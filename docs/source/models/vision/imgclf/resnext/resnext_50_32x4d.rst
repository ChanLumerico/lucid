resnext_50_32x4d
================

.. autofunction:: lucid.models.resnext_50_32x4d

The `resnext_50_32x4d` function constructs a ResNeXt-50 model with 32 groups and
a base width of 4.

**Total Parameters**: 25,028,904

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnext_50_32x4d(num_classes: int = 1000, **kwargs) -> ResNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `ResNeXtConfig`, excluding the preset
  `layers`, `cardinality`, and `base_width` fields.

Returns
-------

- **ResNeXt**:
  A ResNeXt-50 model instance with the preset `(layers=[3, 4, 6, 3], cardinality=32, base_width=4)`.
