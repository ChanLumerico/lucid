convnext_xlarge
===============

.. autofunction:: lucid.models.convnext_xlarge

The `convnext_xlarge` function constructs the ConvNeXt-XLarge preset.
This preset uses the default `ConvNeXtConfig` stage layout for the xlarge variant.

**Total Parameters**: 350,196,968

Function Signature
------------------

.. code-block:: python

    @register_model
    def convnext_xlarge(num_classes: int = 1000, **kwargs) -> ConvNeXt

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
  A ConvNeXt model instance constructed from the xlarge preset config.
