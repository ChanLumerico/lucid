efficientformer_l3
==================

.. autofunction:: lucid.models.efficientformer_l3

The `efficientformer_l3` function constructs the L3 EfficientFormer preset.

**Total Parameters**: 30,893,000

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientformer_l3(num_classes: int = 1000, **kwargs) -> EfficientFormer

Parameters
----------

- **num_classes** (*int*, optional):
  Number of output classes for classification. Default is `1000`.
- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `EfficientFormerConfig`, excluding
  the preset `depths`, `embed_dims`, and `num_vit` fields.

Returns
-------

- **EfficientFormer**:
  An `EfficientFormer` instance constructed from the L3 preset config.

Preset Config
-------------

- **depths**: `(4, 4, 12, 6)`
- **embed_dims**: `(64, 128, 320, 512)`
- **num_vit**: `4`
