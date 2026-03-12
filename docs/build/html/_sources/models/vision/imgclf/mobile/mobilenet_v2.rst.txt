mobilenet_v2
============

.. autofunction:: lucid.models.mobilenet_v2

The `mobilenet_v2` function constructs the standard MobileNet-v2 model.
This preset uses the default `MobileNetV2Config` stage layout.

**Total Parameters**: 3,504,872

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v2(num_classes: int = 1000, **kwargs) -> MobileNet_V2

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `MobileNetV2Config`, excluding the preset
  `stage_configs`, `stem_channels`, and `last_channels` fields.

Returns
-------

- **MobileNet_V2**:
  A MobileNet-v2 model instance constructed from the default preset config.
