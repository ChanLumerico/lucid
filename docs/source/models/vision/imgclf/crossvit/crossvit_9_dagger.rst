crossvit_9_dagger
=================

.. autofunction:: lucid.models.crossvit_9_dagger

The `crossvit_9_dagger` function constructs the dagger 9-layer CrossViT
preset.

**Total Parameters**: 8,776,592

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_9_dagger(num_classes: int = 1000, **kwargs) -> CrossViT

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
  A `CrossViT` instance constructed from the dagger 9-layer preset config.

Preset Config
-------------

- **img_size**: `(240, 224)`
- **patch_size**: `(12, 16)`
- **embed_dim**: `(128, 256)`
- **depth**: `((1, 3, 0), (1, 3, 0), (1, 3, 0))`
- **num_heads**: `(4, 4)`
- **mlp_ratio**: `(3, 3, 1)`
- **qkv_bias**: `True`
- **multi_conv**: `True`
