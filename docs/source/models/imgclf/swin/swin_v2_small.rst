swin_v2_small
=============

.. autofunction:: lucid.models.swin_v2_small

The `swin_v2_small` function instantiates a small-sized Swin Transformer V2 model
with a predefined architecture. It incorporates enhanced normalization and a more
stable shifted window mechanism, making it effective for a wide range of vision
tasks such as classification, object detection, and segmentation.

**Total Parameters**: 49,731,106

Function Signature
------------------

.. code-block:: python

    @register_model
    def swin_v2_small(img_size: int = 224, num_classes: int = 1000, **kwargs) -> SwinTransformer_V2

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
    The common default is `[3, 6, 12, 24]`, allowing the model to effectively
    capture multi-scale information at different depths.

Returns
-------
- **SwinTransformer_V2**:
  An instance of the `SwinTransformer_V2` class configured as a robust vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.swin_v2_small()
    >>> print(model)
    SwinTransformer_V2(img_size=224, num_classes=1000, ...)

