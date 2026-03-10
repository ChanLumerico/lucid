convnext_base
=============

.. autofunction:: lucid.models.convnext_base

The `convnext_base` function constructs the ConvNeXt-Base preset.
This preset uses the default `ConvNeXtConfig` stage layout for the base variant.

**Total Parameters**: 88,591,464

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_base(num_classes: int = 1000, **kwargs) -> ConvNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `ConvNeXtConfig`, excluding the
  preset `depths` and `dims` fields.

Returns
-------

- **ConvNeXt**:
  A ConvNeXt model instance constructed from the base preset config.
