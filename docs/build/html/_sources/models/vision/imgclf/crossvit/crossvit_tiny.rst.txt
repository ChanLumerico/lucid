crossvit_tiny
=============

.. autofunction:: lucid.models.crossvit_tiny

The `crossvit_tiny` function constructs the tiny CrossViT preset.

**Total Parameters**: 7,014,800

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_tiny(num_classes: int = 1000, **kwargs) -> CrossViT

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
  A `CrossViT` instance constructed from the tiny preset config.

Preset Config
-------------

- **img_size**: `(240, 224)`
- **patch_size**: `(12, 16)`
- **embed_dim**: `(96, 192)`
- **depth**: `((1, 4, 0), (1, 4, 0), (1, 4, 0))`
- **num_heads**: `(3, 3)`
- **mlp_ratio**: `(4, 4, 1)`
- **qkv_bias**: `True`
- **multi_conv**: `False`
