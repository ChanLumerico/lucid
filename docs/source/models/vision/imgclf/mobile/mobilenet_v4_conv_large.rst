mobilenet_v4_conv_large
=======================

.. autofunction:: lucid.models.mobilenet_v4_conv_large

The `mobilenet_v4_conv_large` function constructs the MobileNet-v4 Conv Large model.

**Total Parameters**: 32,590,864

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v4_conv_large(num_classes: int = 1000, **kwargs) -> MobileNet_V4

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `MobileNetV4Config`, excluding the preset `cfg` field.

Returns
-------

- **MobileNet_V4**:
  A MobileNet-v4 Conv Large model instance constructed from the preset stage dictionary.
