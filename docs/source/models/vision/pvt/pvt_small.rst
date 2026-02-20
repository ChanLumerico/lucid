pvt_small
=========

.. autofunction:: lucid.models.pvt_small

The `pvt_small` function instantiates the PVT-Small model, a variant of the Pyramid 
Vision Transformer (PVT) that provides a stronger hierarchical feature extraction 
while maintaining efficiency. The model is a great balance between performance and 
computational complexity, designed for more advanced image classification tasks.

**Total Parameters**: 23,003,048

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_small(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

Parameters
----------

- **img_size** (*int*, optional): 
  The input image size. Default is `224`.

- **num_classes** (*int*, optional): 
  The number of output classes for classification. Default is `1000`.

- **kwargs** (*dict*, optional): 
  Additional parameters for customization, including:

  - **embed_dims** (*list[int]*): 
    A list specifying the embedding dimensions at different hierarchical levels. 
    The default configuration for PVT-Small is `[64, 128, 320, 512]`.

  - **depths** (*list[int]*): 
    A list specifying the number of transformer blocks in each stage. 
    The default values for PVT-Small are `[3, 4, 6, 3]`, offering an enhanced 
    feature representation.

  - **num_heads** (*list[int]*): 
    The number of self-attention heads in each transformer stage. 
    The default values are `[1, 2, 5, 8]`, which increase as the model depth progresses.

  - **mlp_ratios** (*list[float]*): 
    The ratio of MLP hidden dimensions to the embedding dimension in each stage. 
    Default values are `[8.0, 8.0, 4.0, 4.0]`.

  - **sr_ratios** (*list[float]*): 
    The spatial reduction ratios used in the self-attention mechanism. 
    Default values are `[8.0, 4.0, 2.0, 1.0]`, which balance between global 
    and local attention.

Returns
-------

- **PVT**: 
  An instance of the `PVT` class configured as a PVT-Small vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.pvt_small()
    >>> print(model)
