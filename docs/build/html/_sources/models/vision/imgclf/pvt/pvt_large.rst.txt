pvt_large
=========

.. autofunction:: lucid.models.pvt_large

The `pvt_large` function constructs the PVT-Large preset.
This preset uses the default `PVTConfig` stage layout for the large variant.

**Total Parameters**: 55,359,848

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_large(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

Parameters
----------

- **img_size** (*int*, optional):
  Input image size. Default is `224`.
- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `PVTConfig`, excluding the preset
  `patch_size`, `embed_dims`, `num_heads`, `mlp_ratios`, `qkv_bias`,
  `norm_layer`, `depths`, and `sr_ratios` fields.

Returns
-------

- **PVT**:
  A PVT model instance constructed from the large preset config.
