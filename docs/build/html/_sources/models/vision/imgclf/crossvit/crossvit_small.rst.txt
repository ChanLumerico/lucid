crossvit_small
==============

.. autofunction:: lucid.models.crossvit_small

The `crossvit_small` function constructs the small CrossViT preset.

**Total Parameters**: 26,856,272

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_small(num_classes: int = 1000, **kwargs) -> CrossViT

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `CrossViTConfig`, excluding the
  preset `img_size`, `patch_size`, `embed_dim`, `depth`, `num_heads`,
  `mlp_ratio`, `qkv_bias`, `norm_layer`, and `multi_conv` fields.

Returns
-------

- **CrossViT**:
  A `CrossViT` instance constructed from the small preset config.

Preset Config
-------------

- **img_size**: `(240, 224)`
- **patch_size**: `(12, 16)`
- **embed_dim**: `(192, 384)`
- **depth**: `((1, 4, 0), (1, 4, 0), (1, 4, 0))`
- **num_heads**: `(6, 6)`
- **mlp_ratio**: `(4, 4, 1)`
- **qkv_bias**: `True`
- **multi_conv**: `False`
