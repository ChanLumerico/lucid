sk_resnet_18
============

.. autofunction:: lucid.models.sk_resnet_18

The `sk_resnet_18` function constructs a Selective Kernel ResNet-18 model.
This preset uses `SKNetConfig(block="basic", layers=[2, 2, 2, 2])`.

**Total Parameters**: 25,647,368

Function Signature
------------------

.. code-block:: python

    @register_model
    def sk_resnet_18(num_classes: int = 1000, **kwargs) -> SKNet

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
  A Selective Kernel ResNet-18 model instance with the preset
  `(block="basic", layers=[2, 2, 2, 2])`.
