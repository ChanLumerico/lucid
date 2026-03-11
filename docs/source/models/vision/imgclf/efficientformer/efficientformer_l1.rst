efficientformer_l1
==================

.. autofunction:: lucid.models.efficientformer_l1

The `efficientformer_l1` function constructs the L1 EfficientFormer preset.

**Total Parameters**: 11,840,928

Function Signature
------------------

.. code-block:: python

    @register_model
    def efficientformer_l1(num_classes: int = 1000, **kwargs) -> EfficientFormer

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
  An `EfficientFormer` instance constructed from the L1 preset config.

Preset Config
-------------

- **depths**: `(3, 2, 6, 4)`
- **embed_dims**: `(48, 96, 224, 448)`
- **num_vit**: `1`
