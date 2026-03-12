vit_tiny
========

.. autofunction:: lucid.models.vit_tiny

The `vit_tiny` function instantiates a small Vision Transformer (ViT-Ti) model 
with a predefined architecture. This preset uses the default `ViTConfig`
transformer dimensions for the tiny variant.

**Total Parameters**: 5,717,416

Function Signature
------------------

.. code-block:: python

    @register_model
    def vit_tiny(
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
  A ViT model instance constructed from the tiny preset config.
