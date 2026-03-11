pvt_v2_b5
=========

.. autofunction:: lucid.models.pvt_v2_b5

The `pvt_v2_b5` function constructs the PVT-v2-B5 preset.
This preset uses the default `PVTV2Config` stage layout for the B5 variant.

**Total Parameters**: 82,882,984

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_v2_b5(img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PVT_V2

Parameters
----------

- **img_size** (*int*, optional):
  Input image size. Default is `224`.
- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **in_channels** (*int*, optional):
  Number of input image channels. Default is `3`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `PVTV2Config`, excluding the preset
  `patch_size`, `embed_dims`, `num_heads`, `mlp_ratios`, `qkv_bias`,
  `norm_layer`, `depths`, `sr_ratios`, and `num_stages` fields.

Returns
-------

- **PVT_V2**:
  A PVT-v2 model instance constructed from the B5 preset config.
