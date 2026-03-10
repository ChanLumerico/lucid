mobilenet_v4_conv_small
=======================

.. autofunction:: lucid.models.mobilenet_v4_conv_small

The `mobilenet_v4_conv_small` function constructs the MobileNet-v4 Conv Small model.

**Total Parameters**: 3,774,024

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v4_conv_small(num_classes: int = 1000, **kwargs) -> MobileNet_V4

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `MobileNetV4Config`, excluding the preset `cfg` field.

Returns
-------

- **MobileNet_V4**:
  A MobileNet-v4 Conv Small model instance constructed from the preset stage dictionary.
