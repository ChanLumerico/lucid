vit_huge
========

.. autofunction:: lucid.models.vit_huge

The `vit_huge` function instantiates a Vision Transformer (ViT-H) model
with a predefined architecture. This preset uses the default `ViTConfig`
transformer dimensions for the huge variant.

**Total Parameters**: 632,199,400

Function Signature
------------------

.. code-block:: python

    @register_model
    def vit_huge(
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        **kwargs
    ) -> ViT

Parameters
----------
- **image_size** (*int*, optional):
  The size of the input image (assumes square images). Default is 224.

- **patch_size** (*int*, optional):
  The size of the patches the image is divided into. Default is 16.

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments forwarded to `ViTConfig`, excluding the preset
  `embedding_dim`, `depth`, `num_heads`, and `mlp_dim` fields.

Returns
-------
- **ViT**:
  A ViT model instance constructed from the huge preset config.
