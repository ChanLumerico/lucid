crossvit_base
=============

.. autofunction:: lucid.models.crossvit_base

The `crossvit_base` function constructs the base CrossViT preset.

**Total Parameters**: 105,025,232

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_base(num_classes: int = 1000, **kwargs) -> CrossViT

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
  A `CrossViT` instance constructed from the base preset config.

Preset Config
-------------

- **img_size**: `(240, 224)`
- **patch_size**: `(12, 16)`
- **embed_dim**: `(384, 768)`
- **depth**: `((1, 4, 0), (1, 4, 0), (1, 4, 0))`
- **num_heads**: `(12, 12)`
- **mlp_ratio**: `(4, 4, 1)`
- **qkv_bias**: `True`
- **multi_conv**: `False`
