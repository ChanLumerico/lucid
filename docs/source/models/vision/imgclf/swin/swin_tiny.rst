swin_tiny
=========

.. autofunction:: lucid.models.swin_tiny

The `swin_tiny` function constructs the Swin-Tiny preset.
This preset uses the default `SwinTransformerConfig` stage layout for the tiny variant.

**Total Parameters**: 28,288,354

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_tiny(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer

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
  A Swin Transformer model instance constructed from the tiny preset config.
