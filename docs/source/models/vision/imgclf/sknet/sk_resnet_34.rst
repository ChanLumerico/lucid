sk_resnet_34
============

.. autofunction:: lucid.models.sk_resnet_34

The `sk_resnet_34` function constructs a Selective Kernel ResNet-34 model.
This preset uses `SKNetConfig(block="basic", layers=[3, 4, 6, 3])`.

**Total Parameters**: 45,895,512

Function Signature
------------------

.. code-block:: python

    @register_model
    def sk_resnet_34(num_classes: int = 1000, **kwargs) -> SKNet

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
  A Selective Kernel ResNet-34 model instance with the preset
  `(block="basic", layers=[3, 4, 6, 3])`.
