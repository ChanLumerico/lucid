convnext_large
==============

.. autofunction:: lucid.models.convnext_large

The `convnext_large` function constructs the ConvNeXt-Large preset.
This preset uses the default `ConvNeXtConfig` stage layout for the large variant.

**Total Parameters**: 197,767,336

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_large(num_classes: int = 1000, **kwargs) -> ConvNeXt

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
  A ConvNeXt model instance constructed from the large preset config.
