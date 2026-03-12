inception_next_tiny
===================

.. autofunction:: lucid.models.inception_next_tiny

The `inception_next_tiny` function constructs the InceptionNeXt-Tiny preset.
This preset uses the default `InceptionNeXtConfig` stage layout for the tiny variant.

**Total Parameters**: 28,083,832

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_tiny(num_classes: int = 1000, **kwargs) -> InceptionNeXt

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
  An InceptionNeXt model instance constructed from the tiny preset config.
