swin_v2_huge
============

.. autofunction:: lucid.models.swin_v2_huge

The `swin_v2_huge` function initializes an extra-large Swin Transformer V2 model
with a predefined architecture. This model extends the Swin Transformer V2 family
by incorporating deeper layers, a larger number of attention heads, and more advanced
normalization techniques. It is designed for high-performance vision tasks such as
image classification, object detection, and segmentation.

**Total Parameters**: 657,796,668

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_v2_huge(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer_V2

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
    The typical default configuration is `[2, 2, 18, 2]`, indicating
    that the model has 4 stages with 2, 2, 18, and 2 blocks respectively.

  - **num_heads** (*list[int]*):
    A list specifying the number of attention heads in each stage.
    The common default is `[6, 12, 24, 48]`, allowing for significantly improved
    multi-scale feature extraction and global context understanding.

Returns
-------
- **SwinTransformer_V2**:
  An instance of the `SwinTransformer_V2` class configured as a high-capacity 
  vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.swin_v2_huge()
    >>> print(model)
    SwinTransformer_V2(img_size=224, num_classes=1000, ...)

