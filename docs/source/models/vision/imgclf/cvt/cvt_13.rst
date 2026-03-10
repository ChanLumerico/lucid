cvt_13
======

.. autofunction:: lucid.models.cvt_13

The `cvt_13` function constructs the CvT-13 preset.
This preset uses the default `CvTConfig` stage layout for the 13-layer variant.

**Total Parameters**: 19,997,480

Function Signature
------------------

.. code-block:: python

    @register_model
    def cvt_13(num_classes: int = 1000, **kwargs) -> CvT

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `CvTConfig`, excluding the preset
  `num_stages`, `patch_size`, `patch_stride`, `patch_padding`, `dim_embed`,
  `num_heads`, and `depth` fields.

Returns
-------

- **CvT**:
  A CvT model instance constructed from the CvT-13 preset config.
