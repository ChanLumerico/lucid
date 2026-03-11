efficientnet_v2_s
=================

.. autofunction:: lucid.models.efficientnet_v2_s

The `efficientnet_v2_s` function instantiates a small variant of the EfficientNet-v2 model, 
specifically designed for lightweight tasks while maintaining high performance.

**Total Parameters**: 21,136,440

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientnet_v2_s(num_classes: int = 1000, **kwargs) -> EfficientNet_V2:

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
  An instance of the `EfficientNet_V2` model pre-configured for the small variant.
