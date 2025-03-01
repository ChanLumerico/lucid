swin_v2_giant
=============

.. autofunction:: lucid.models.swin_v2_giant

The `swin_v2_giant` function initializes the largest variant of the Swin Transformer V2 model,
optimized for extremely high-performance vision tasks such as large-scale image classification,
object detection, and segmentation. This model is designed to handle complex data distributions
by leveraging deeper layers, a higher number of attention heads, and more efficient normalization.

**Total Parameters**: 3,000,869,564

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_v2_giant(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer_V2

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
    The typical default configuration is `[2, 2, 42, 4]`, indicating
    that the model has 4 stages with 2, 2, 18, and 2 blocks respectively.

  - **num_heads** (*list[int]*):
    A list specifying the number of attention heads in each stage.
    The common default is `[6, 12, 24, 48]`, allowing the model to efficiently
    capture global dependencies and perform detailed feature extraction.

Returns
-------
- **SwinTransformer_V2**:
  An instance of the `SwinTransformer_V2` class configured as an ultra-large 
  vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.swin_v2_giant()
    >>> print(model)
    SwinTransformer_V2(img_size=224, num_classes=1000, ...)

