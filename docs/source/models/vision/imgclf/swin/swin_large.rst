swin_large
==========

.. autofunction:: lucid.models.swin_large

The `swin_large` function constructs the Swin-Large preset.
This preset uses the default `SwinTransformerConfig` stage layout for the large variant.

**Total Parameters**: 196,532,476

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_large(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer

Parameters
----------

- **img_size** (*int*, optional):
  Input image size. Default is `224`.
- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `SwinTransformerConfig`, excluding
  the preset `embed_dim`, `depths`, and `num_heads` fields.

Returns
-------

- **SwinTransformer**:
  A Swin Transformer model instance constructed from the large preset config.
