efficientformer_l7
==================

.. autofunction:: lucid.models.efficientformer_l7

The `efficientformer_l7` function constructs the L7 EfficientFormer preset.

**Total Parameters**: 81,460,328

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientformer_l7(num_classes: int = 1000, **kwargs) -> EfficientFormer

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
  An `EfficientFormer` instance constructed from the L7 preset config.

Preset Config
-------------

- **depths**: `(6, 6, 18, 8)`
- **embed_dims**: `(96, 192, 384, 768)`
- **num_vit**: `8`
