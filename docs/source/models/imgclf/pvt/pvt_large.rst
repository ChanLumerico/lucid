pvt_large
=========

.. autofunction:: lucid.models.pvt_large

The `pvt_large` function instantiates the PVT-Large model, a more advanced version 
of the Pyramid Vision Transformer (PVT). PVT-Large is intended for high-performance 
vision tasks, combining high depth and feature complexity with excellent efficiency.

**Total Parameters**: 55,359,848

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_large(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

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
    The default for PVT-Large is `[64, 128, 320, 512]`, 
    ensuring high feature representation.

  - **depths** (*list[int]*): 
    A list specifying the number of transformer blocks in each stage. 
    The default configuration for PVT-Large is `[3, 8, 27, 3]`, 
    reflecting an increased depth for complex features.

  - **num_heads** (*list[int]*): 
    The number of self-attention heads in each transformer stage. 
    The default values are `[1, 2, 5, 8]`, progressively increasing the model's 
    capacity to capture long-range dependencies.

  - **mlp_ratios** (*list[float]*): 
    The ratio of MLP hidden dimensions to the embedding dimension in each stage. 
    Default values are `[8.0, 8.0, 4.0, 4.0]`.

  - **sr_ratios** (*list[float]*): 
    The spatial reduction ratios used in the self-attention mechanism. 
    Default values are `[8.0, 4.0, 2.0, 1.0]`.

Returns
-------

- **PVT**: 
  An instance of the `PVT` class configured as a PVT-Large vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.pvt_large()
    >>> print(model)
