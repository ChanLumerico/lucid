convnext_v2_base
================

.. autofunction:: lucid.models.convnext_v2_base

The `convnext_v2_base` function constructs the ConvNeXt-v2 Base preset.
This preset uses the default `ConvNeXtV2Config` stage layout for the base variant.

**Total Parameters**: 88,717,800

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_v2_base(num_classes: int = 1000, **kwargs) -> ConvNeXt_V2

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
  A ConvNeXt-v2 model instance constructed from the base preset config.
