inception_next_base
===================

.. autofunction:: lucid.models.inception_next_base

The `inception_next_base` function constructs the InceptionNeXt-Base preset.
This preset uses the default `InceptionNeXtConfig` stage layout for the base variant.

**Total Parameters**: 86,748,840

Function Signature
------------------

.. code-block:: python

    @register_model
    def inception_next_base(num_classes: int = 1000, **kwargs) -> InceptionNeXt

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
  An InceptionNeXt model instance constructed from the base preset config.
