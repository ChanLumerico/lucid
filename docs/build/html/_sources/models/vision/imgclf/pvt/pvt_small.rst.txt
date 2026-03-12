pvt_small
=========

.. autofunction:: lucid.models.pvt_small

The `pvt_small` function constructs the PVT-Small preset.
This preset uses the default `PVTConfig` stage layout for the small variant.

**Total Parameters**: 23,003,048

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_small(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

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
  A PVT model instance constructed from the small preset config.
