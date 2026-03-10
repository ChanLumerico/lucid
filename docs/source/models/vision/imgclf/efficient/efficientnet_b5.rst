efficientnet_b5
===============

.. autofunction:: lucid.models.efficientnet_b5

The `efficientnet_b5` function creates an instance of the `EfficientNet-B5` model, 
a lightweight and efficient convolutional neural network preset built from the
default `EfficientNetConfig` values for EfficientNet-B5.

**Total Parameters**: 30,393,432

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientnet_b5(num_classes: int = 1000, **kwargs) -> EfficientNet:

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
   An instance of the `EfficientNet` class constructed from the EfficientNet-B5 preset.
