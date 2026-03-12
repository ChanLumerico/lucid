efficientnet_b6
===============

.. autofunction:: lucid.models.efficientnet_b6

The `efficientnet_b6` function creates an instance of the `EfficientNet-B6` model, 
a lightweight and efficient convolutional neural network preset built from the
default `EfficientNetConfig` values for EfficientNet-B6.

**Total Parameters**: 43,046,128

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientnet_b6(num_classes: int = 1000, **kwargs) -> EfficientNet:

Parameters
----------

- **num_classes** (*int*, optional):
   The number of output classes for the classification task. 
   Defaults to 1000 (e.g., for ImageNet).

- **kwargs** (*dict*, optional):
   Additional keyword arguments forwarded to `EfficientNetConfig`, excluding the
   preset `width_coef`, `depth_coef`, `scale`, and `dropout` fields.

Returns
-------

- **EfficientNet**:
   An instance of the `EfficientNet` class constructed from the EfficientNet-B6 preset.
