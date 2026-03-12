swin_v2_tiny
============

.. autofunction:: lucid.models.swin_v2_tiny

The `swin_v2_tiny` function constructs the Swin V2-Tiny preset.
This preset uses the default `SwinTransformerV2Config` stage layout for the tiny variant.

**Total Parameters**: 28,349,842

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_v2_tiny(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer_V2

Parameters
----------

- **img_size** (*int*, optional):
  Input image size. Default is `224`.
- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `SwinTransformerV2Config`, excluding
  the preset `embed_dim`, `depths`, and `num_heads` fields.

Returns
-------

- **SwinTransformer_V2**:
  A Swin Transformer V2 model instance constructed from the tiny preset config.
