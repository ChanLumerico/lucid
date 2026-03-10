mobilenet_v4_hybrid_medium
==========================

.. autofunction:: lucid.models.mobilenet_v4_hybrid_medium

The `mobilenet_v4_hybrid_medium` function constructs the MobileNet-v4 Hybrid Medium model.

**Total Parameters**: 11,070,136

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v4_hybrid_medium(num_classes: int = 1000, **kwargs) -> MobileNet_V4

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `MobileNetV4Config`, excluding the preset `cfg` field.

Returns
-------

- **MobileNet_V4**:
  A MobileNet-v4 Hybrid Medium model instance constructed from the preset stage dictionary.
