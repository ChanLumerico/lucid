cvt_w24
=======

.. autofunction:: lucid.models.cvt_w24

The `cvt_w24` function instantiates a CvT-W24 model, a variant of the Convolutional 
Vision Transformer (CvT). This model integrates depthwise convolutions into the 
self-attention mechanism, enhancing local feature extraction while maintaining 
global contextual understanding. The CvT-W24 architecture is designed to offer 
high performance in image classification tasks, with a larger scale and more 
transformer blocks compared to smaller variants.

**Total Parameters**: 277,196,392

Function Signature
------------------

.. code-block:: python

    @register_model
    def cvt_w24(num_classes: int = 1000, **kwargs) -> CvT

Parameters
----------

- **num_classes** (*int*, optional):
  The number of output classes for classification. Default is 1000.

- **kwargs** (*dict*, optional):
  Additional parameters for customization, including:

  - **embed_dim** (*list[int]*):
    A list specifying the embedding dimensions for different stages.
    The default for CvT-W24 is `[192, 768, 1024]`, indicating the number of channels
    at each stage of the model.

  - **depths** (*list[int]*):
    A list specifying the number of transformer blocks in each stage.
    The default configuration for CvT-W24 is `[2, 2, 20]`, representing
    the depth of the transformer at each stage.

  - **num_heads** (*list[int]*):
    A list specifying the number of attention heads in each stage.
    The default values for CvT-W24 are `[3, 12, 16]`, enabling multi-head
    self-attention at different scales.

Returns
-------
- **CvT**:
  An instance of the `CvT` class configured as a CvT-W24 vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.cvt_w24()
    >>> print(model)
    CvT(num_classes=1000, embed_dim=[192, 768, 1024], depths=[2, 2, 20], num_heads=[3, 12, 16], ...)
