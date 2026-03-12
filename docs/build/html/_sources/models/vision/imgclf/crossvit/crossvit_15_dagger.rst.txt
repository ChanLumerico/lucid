crossvit_15_dagger
==================

.. autofunction:: lucid.models.crossvit_15_dagger

The `crossvit_15_dagger` function constructs the dagger 15-layer CrossViT
preset.

**Total Parameters**: 28,209,008

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_15_dagger(num_classes: int = 1000, **kwargs) -> CrossViT

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
  A `CrossViT` instance constructed from the dagger 15-layer preset config.

Preset Config
-------------

- **img_size**: `(240, 224)`
- **patch_size**: `(12, 16)`
- **embed_dim**: `(192, 384)`
- **depth**: `((1, 5, 0), (1, 5, 0), (1, 5, 0))`
- **num_heads**: `(6, 6)`
- **mlp_ratio**: `(3, 3, 1)`
- **qkv_bias**: `True`
- **multi_conv**: `True`
