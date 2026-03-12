convnext_small
==============

.. autofunction:: lucid.models.convnext_small

The `convnext_small` function constructs the ConvNeXt-Small preset.
This preset uses the default `ConvNeXtConfig` stage layout for the small variant.

**Total Parameters**: 46,884,148

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_small(num_classes: int = 1000, **kwargs) -> ConvNeXt

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
  A ConvNeXt model instance constructed from the small preset config.
