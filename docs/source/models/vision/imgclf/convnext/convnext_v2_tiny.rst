convnext_v2_tiny
================

.. autofunction:: lucid.models.convnext_v2_tiny

The `convnext_v2_tiny` function constructs the ConvNeXt-v2 Tiny preset.
This preset uses the default `ConvNeXtV2Config` stage layout for the tiny variant.

**Total Parameters**: 28,635,496

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

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
  A ConvNeXt-v2 model instance constructed from the tiny preset config.
