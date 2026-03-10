csp_resnext_50_32x4d
====================

.. autofunction:: lucid.models.csp_resnext_50_32x4d

The `csp_resnext_50_32x4d` function constructs the CSPResNeXt-50 32x4d preset.
This preset uses the default `CSPNetConfig` stage layout for the ResNeXt-style variant.

**Total Parameters**: 22,509,864

Function Signature
------------------

.. code-block:: python

    @register_model
    def csp_resnext_50_32x4d(
        num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 64, **kwargs
    ) -> CSPNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **split_ratio** (*float*, optional):
  CSP split ratio. Default is `0.5`.
- **stem_channels** (*int*, optional):
  Stem output width. Default is `64`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `CSPNetConfig`, excluding the
  preset `stage_specs`, `stack_type`, `groups`, `base_width`, and `feature_channels` fields.

Returns
-------

- **CSPNet**:
  A CSPNet model instance constructed from the CSPResNeXt-50 preset config.
