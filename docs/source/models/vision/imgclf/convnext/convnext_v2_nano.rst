convnext_v2_nano
================

.. autofunction:: lucid.models.convnext_v2_nano

The `convnext_v2_nano` function constructs the ConvNeXt-v2 Nano preset.
This preset uses the default `ConvNeXtV2Config` stage layout for the nano variant.

**Total Parameters**: 15,623,800

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_nano(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `ConvNeXtV2Config`, excluding the
  preset `depths` and `dims` fields.

Returns
-------

- **ConvNeXt_V2**:
  A ConvNeXt-v2 model instance constructed from the nano preset config.
