csp_darknet_53
==============

.. autofunction:: lucid.models.csp_darknet_53

The `csp_darknet_53` function constructs the CSPDarknet-53 preset.
This preset uses the default `CSPNetConfig` stage layout for the Darknet-style variant.

**Total Parameters**: 27,278,536

Function Signature
------------------

.. code-block:: python

    @register_model
    def csp_darknet_53(
        num_classes: int = 1000, split_ratio: float = 0.5, stem_channels: int = 32, **kwargs
    ) -> CSPNet

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **split_ratio** (*float*, optional):
  CSP split ratio. Default is `0.5`.
- **stem_channels** (*int*, optional):
  Stem output width. Default is `32`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `CSPNetConfig`, excluding the
  preset `stage_specs`, `stack_type`, `feature_channels`, and `pre_kernel_size` fields.

Returns
-------

- **CSPNet**:
  A CSPNet model instance constructed from the CSPDarknet-53 preset config.
