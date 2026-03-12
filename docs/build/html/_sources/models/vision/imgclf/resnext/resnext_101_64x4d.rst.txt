resnext_101_64x4d
=================

.. autofunction:: lucid.models.resnext_101_64x4d

The `resnext_101_64x4d` function constructs a ResNeXt-101 model with 64 groups and
a base width of 4.

**Total Parameters**: 83,455,272

Function Signature
------------------

.. code-block:: python

    @register_model
    def resnext_101_64x4d(num_classes: int = 1000, **kwargs) -> ResNeXt

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
  A ResNeXt-101 model instance with the preset `(layers=[3, 4, 23, 3], cardinality=64, base_width=4)`.
