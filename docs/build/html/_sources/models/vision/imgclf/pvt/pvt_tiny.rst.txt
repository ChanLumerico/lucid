pvt_tiny
========

.. autofunction:: lucid.models.pvt_tiny

The `pvt_tiny` function constructs the PVT-Tiny preset.
This preset uses the default `PVTConfig` stage layout for the tiny variant.

**Total Parameters**: 12,457,192

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_tiny(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

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
  A PVT model instance constructed from the tiny preset config.
