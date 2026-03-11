efficientnet_b3
===============

.. autofunction:: lucid.models.efficientnet_b3

The `efficientnet_b3` function creates an instance of the `EfficientNet-B3` model, 
a lightweight and efficient convolutional neural network preset built from the
default `EfficientNetConfig` values for EfficientNet-B3.

**Total Parameters**: 12,235,536

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientnet_b3(num_classes: int = 1000, **kwargs) -> EfficientNet:

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
   An instance of the `EfficientNet` class constructed from the EfficientNet-B3 preset.
