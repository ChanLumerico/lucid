vit_tiny
========

.. autofunction:: lucid.models.vit_tiny

The `vit_tiny` function instantiates a small Vision Transformer (ViT-Ti) model 
with a predefined architecture. This model is useful for lightweight applications 
requiring a Transformer-based vision model.

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
  Additional parameters for customization.

Returns
-------
- **ViT**:
  An instance of the `ViT` class configured as a lightweight vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.vit_tiny()
    >>> print(model)
    ViT(img_size=224, patch_size=16, num_classes=1000, ...)
