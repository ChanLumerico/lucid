crossvit_18
===========

.. autofunction:: lucid.models.crossvit_18

The `crossvit_18` function constructs the 18-layer CrossViT preset.

**Total Parameters**: 43,271,408

Function Signature
------------------

.. code-block:: python

    @register_model
    def crossvit_18(num_classes: int = 1000, **kwargs) -> CrossViT

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
  A `CrossViT` instance constructed from the 18-layer preset config.

Preset Config
-------------

- **img_size**: `(240, 224)`
- **patch_size**: `(12, 16)`
- **embed_dim**: `(224, 448)`
- **depth**: `((1, 6, 0), (1, 6, 0), (1, 6, 0))`
- **num_heads**: `(7, 7)`
- **mlp_ratio**: `(3, 3, 1)`
- **qkv_bias**: `True`
- **multi_conv**: `False`
