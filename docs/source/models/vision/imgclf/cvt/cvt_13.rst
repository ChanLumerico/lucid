cvt_13
======

.. autofunction:: lucid.models.cvt_13

The `cvt_13` function instantiates a CvT-13 model, a variant of the Convolutional 
Vision Transformer (CvT). This model integrates depthwise convolutions into the 
self-attention mechanism, enhancing local feature extraction while maintaining 
global contextual understanding. The CvT-13 architecture is designed to balance 
efficiency and performance for image classification tasks.

**Total Parameters**: 19,997,480

Function Signature
------------------

.. code-block:: python

    @register_model
    def cvt_13(num_classes: int = 1000, **kwargs) -> CvT

Parameters
----------
- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional parameters for customization, including:

  - **embed_dim** (*list[int]*):
    A list specifying the embedding dimensions for different stages.
    The default for CvT-13 is `[64, 192, 384]`, indicating the number of channels
    at each stage of the model.

  - **depths** (*list[int]*):
    A list specifying the number of transformer blocks in each stage.
    The typical default configuration for CvT-13 is `[1, 2, 10]`, representing
    the depth of the transformer at each stage.

  - **num_heads** (*list[int]*):
    A list specifying the number of attention heads in each stage.
    The default values for CvT-13 are `[1, 3, 6]`, enabling multi-head
    self-attention at different scales.

Returns
-------
- **CvT**:
  An instance of the `CvT` class configured as a CvT-13 vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.cvt_13()
    >>> print(model)
    CvT(num_classes=1000, embed_dim=[64, 192, 384], depths=[1, 2, 10], num_heads=[1, 3, 6], ...)

