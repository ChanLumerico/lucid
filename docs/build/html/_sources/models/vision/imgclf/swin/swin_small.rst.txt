swin_small
==========

.. autofunction:: lucid.models.swin_small

The `swin_small` function constructs the Swin-Small preset.
This preset uses the default `SwinTransformerConfig` stage layout for the small variant.

**Total Parameters**: 49,606,258

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_small(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer

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
  A Swin Transformer model instance constructed from the small preset config.
