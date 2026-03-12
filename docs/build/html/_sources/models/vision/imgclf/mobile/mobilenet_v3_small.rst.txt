mobilenet_v3_small
==================

.. autofunction:: lucid.models.mobilenet_v3_small

The `mobilenet_v3_small` function constructs the MobileNet-v3 Small model.
This preset uses the standard MobileNet-v3 Small bottleneck layout and `last_channels=1024`.

**Total Parameters**: 2,537,238

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v3_small(num_classes: int = 1000, **kwargs) -> MobileNet_V3

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
  A MobileNet-v3 Small model instance constructed from the default preset config.
