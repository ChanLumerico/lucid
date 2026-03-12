efficientnet_v2_xl
==================

.. autofunction:: lucid.models.efficientnet_v2_xl

The `efficientnet_v2_xl` function instantiates an extra-large variant of the 
EfficientNet-v2 model, specifically designed for lightweight tasks while maintaining 
high performance.

**Total Parameters**: 210,221,568

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientnet_v2_xl(num_classes: int = 1000, **kwargs) -> EfficientNet_V2:

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for the model. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `EfficientNetV2Config`, excluding
  the preset `block_cfg`, `dropout`, and `drop_path_rate` fields.

Returns
-------

- **EfficientNet_V2**: 
  An instance of the `EfficientNet_V2` model pre-configured for the extra-large variant.
