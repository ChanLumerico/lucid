vit_huge
========

.. autofunction:: lucid.models.vit_huge

The `vit_huge` function instantiates a Vision Transformer (ViT-H) model
with a predefined architecture. This model provides state-of-the-art accuracy
and is designed for large-scale vision tasks with extreme computational power.

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
  Additional parameters for customization.

Returns
-------
- **ViT**:
  An instance of the `ViT` class configured as a high-capacity vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.vit_huge()
    >>> print(model)
    ViT(img_size=224, patch_size=16, num_classes=1000, ...)

