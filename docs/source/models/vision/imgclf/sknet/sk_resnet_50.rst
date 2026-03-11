sk_resnet_50
============

.. autofunction:: lucid.models.sk_resnet_50

The `sk_resnet_50` function constructs a Selective Kernel ResNet-50 model.
This preset uses `SKNetConfig(block="bottleneck", layers=[3, 4, 6, 3])`.

**Total Parameters**: 57,073,368

Function Signature
------------------

.. code-block:: python

    @register_model
    def sk_resnet_50(num_classes: int = 1000, **kwargs) -> SKNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default is 1000.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `SKNetConfig`, excluding the preset
  `block` and `layers` fields.

Returns
-------

- **SKNet**:
  A Selective Kernel ResNet-50 model instance with the preset
  `(block="bottleneck", layers=[3, 4, 6, 3])`.
