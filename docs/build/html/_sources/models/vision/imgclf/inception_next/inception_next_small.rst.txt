inception_next_small
====================

.. autofunction:: lucid.models.inception_next_small

The `inception_next_small` function constructs the InceptionNeXt-Small preset.
This preset uses the default `InceptionNeXtConfig` stage layout for the small variant.

**Total Parameters**: 49,431,544

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_small(num_classes: int = 1000, **kwargs) -> InceptionNeXt

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `InceptionNeXtConfig`, excluding the
  preset `depths`, `dims`, and `token_mixers` fields.

Returns
-------

- **InceptionNeXt**:
  An InceptionNeXt model instance constructed from the small preset config.
