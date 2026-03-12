pvt_huge
========

.. autofunction:: lucid.models.pvt_huge

The `pvt_huge` function constructs the PVT-Huge preset.
This preset uses the default `PVTConfig` stage layout for the huge variant.

**Total Parameters**: 286,706,920

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_huge(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

Parameters
----------

- **img_size** (*int*, optional):
  Input image size. Default is `224`.
- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `PVTConfig`, excluding the preset
  `patch_size`, `embed_dims`, `num_heads`, `mlp_ratios`, `qkv_bias`,
  `norm_layer`, `depths`, `sr_ratios`, and `drop_path_rate` fields.

Returns
-------

- **PVT**:
  A PVT model instance constructed from the huge preset config.
