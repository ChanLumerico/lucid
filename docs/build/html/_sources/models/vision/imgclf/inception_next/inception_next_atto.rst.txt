inception_next_atto
===================

.. autofunction:: lucid.models.inception_next_atto

The `inception_next_atto` function constructs the InceptionNeXt-Atto preset.
This preset uses the default `InceptionNeXtConfig` stage layout for the atto variant.

**Total Parameters**: 4,156,520

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_atto(num_classes: int = 1000, **kwargs) -> InceptionNeXt

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
  An InceptionNeXt model instance constructed from the atto preset config.
