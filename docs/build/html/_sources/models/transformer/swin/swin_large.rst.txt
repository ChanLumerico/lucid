swin_large
==========

.. autofunction:: lucid.models.swin_large

The `swin_large` function instantiates a large Swin Transformer model with 
a predefined architecture. This model leverages the shifted window mechanism 
to efficiently capture both local and global dependencies, making it suitable 
for image recognition and dense prediction tasks.

**Total Parameters**: 196,532,476

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_large(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer

Parameters
----------
- **img_size** (*int*, optional):
  The size of the input image (assumes square images). Default is 224.

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional parameters for customization, including:

  - **embed_dim** (*int*):
    The dimension of the embedding for the first stage.
    The typical default for the large model is 192.

  - **depths** (*list[int]*):
    A list specifying the number of transformer blocks in each stage.
    The typical default configuration is `[2, 2, 18, 2]`, indicating 
    that the model has 4 stages with 2, 2, 18, and 2 blocks respectively.

  - **num_heads** (*list[int]*):
    A list specifying the number of attention heads in each stage.
    The common default for the large model is `[6, 12, 24, 48]`, corresponding 
    to the number of heads used in each stage, which enables the model to capture 
    multi-scale contextual information.

Returns
-------
- **SwinTransformer**:
  An instance of the `SwinTransformer` class configured as a large vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.swin_large()
    >>> print(model)
    SwinTransformer(img_size=224, num_classes=1000, embed_dim=192, ...)
