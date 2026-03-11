cvt_w24
=======

.. autofunction:: lucid.models.cvt_w24

The `cvt_w24` function constructs the CvT-W24 preset.
This preset uses the default `CvTConfig` stage layout for the wide 24-layer variant.

**Total Parameters**: 277,196,392

Function Signature
------------------

.. code-block:: python

    @register_model
    def cvt_w24(num_classes: int = 1000, **kwargs) -> CvT

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
  A CvT model instance constructed from the CvT-W24 preset config.
