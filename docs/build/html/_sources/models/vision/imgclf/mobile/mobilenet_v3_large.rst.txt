mobilenet_v3_large
==================

.. autofunction:: lucid.models.mobilenet_v3_large

The `mobilenet_v3_large` function constructs the MobileNet-v3 Large model.
This preset uses the standard MobileNet-v3 Large bottleneck layout and `last_channels=1280`.

**Total Parameters**: 5,481,198

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v3_large(num_classes: int = 1000, **kwargs) -> MobileNet_V3

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `MobileNetV3Config`, excluding the preset
  `bottleneck_cfg`, `last_channels`, and `stem_channels` fields.

Returns
-------

- **MobileNet_V3**:
  A MobileNet-v3 Large model instance constructed from the default preset config.
