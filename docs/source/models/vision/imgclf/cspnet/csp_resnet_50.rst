csp_resnet_50
=============

.. autofunction:: lucid.models.csp_resnet_50

The `csp_resnet_50` function constructs the CSPResNet-50 preset.
This preset uses the default `CSPNetConfig` stage layout for the ResNet-style variant.

**Total Parameters**: 22,463,016

Function Signature
------------------

.. code-block:: python

    @register_model
    def csp_resnet_50(
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
  A CSPNet model instance constructed from the CSPResNet-50 preset config.
