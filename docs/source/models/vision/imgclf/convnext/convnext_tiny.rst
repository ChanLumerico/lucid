convnext_tiny
=============

.. autofunction:: lucid.models.convnext_tiny

The `convnext_tiny` function constructs the ConvNeXt-Tiny preset.
This preset uses the default `ConvNeXtConfig` stage layout for the tiny variant.

**Total Parameters**: 28,589,128

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_tiny(num_classes: int = 1000, **kwargs) -> ConvNeXt

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
  A ConvNeXt model instance constructed from the tiny preset config.
