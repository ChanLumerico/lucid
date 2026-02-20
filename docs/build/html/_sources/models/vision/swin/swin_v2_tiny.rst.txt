swin_v2_tiny
============

.. autofunction:: lucid.models.swin_v2_tiny

The `swin_v2_tiny` function instantiates a small Swin Transformer V2 model with a
predefined architecture. This model builds upon the original Swin Transformer
by introducing improved normalization techniques and more robust window attention,
making it well-suited for vision tasks such as image recognition and segmentation.

**Total Parameters**: 28,349,842

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_v2_tiny(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer_V2

Parameters
----------
- **img_size** (*int*, optional):
  The size of the input image (assumes square images). Default is 224.

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional parameters for customization, including:

  - **depths** (*list[int]*):
    A list specifying the number of transformer blocks in each stage.
    The typical default configuration is `[2, 2, 6, 2]`, indicating
    that the model has 4 stages with 2, 2, 6, and 2 blocks respectively.

  - **num_heads** (*list[int]*):
    A list specifying the number of attention heads in each stage.
    The common default is `[3, 6, 12, 24]`, corresponding to the number
    of heads used in each stage, enabling multi-scale feature extraction.

Returns
-------
- **SwinTransformer_V2**:
  An instance of the `SwinTransformer_V2` class configured as a lightweight 
  vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.swin_v2_tiny()
    >>> print(model)
    SwinTransformer_V2(img_size=224, num_classes=1000, ...)

