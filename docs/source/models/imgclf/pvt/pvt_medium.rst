pvt_medium
==========

.. autofunction:: lucid.models.pvt_medium

The `pvt_medium` function instantiates the PVT-Medium model, a more powerful 
variant of the Pyramid Vision Transformer (PVT). It features greater depth and 
complexity, offering strong performance for difficult image recognition tasks 
while maintaining efficiency.

**Total Parameters**: 41,492,648

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_medium(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

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
    The default for PVT-Medium is `[64, 128, 320, 512]`, 
    offering deeper feature representations.

  - **depths** (*list[int]*): 
    A list specifying the number of transformer blocks in each stage. 
    The default values for PVT-Medium are `[3, 4, 18, 3]`, 
    providing deeper levels of abstraction.

  - **num_heads** (*list[int]*): 
    The number of self-attention heads in each transformer stage. 
    The default values are `[1, 2, 5, 8]`, progressively 
    capturing more complex relationships.

  - **mlp_ratios** (*list[float]*): 
    The ratio of MLP hidden dimensions to the embedding dimension in each stage. 
    Default values are `[8.0, 8.0, 4.0, 4.0]`.

  - **sr_ratios** (*list[float]*): 
    The spatial reduction ratios used in the self-attention mechanism. 
    Default values are `[8.0, 4.0, 2.0, 1.0]`.

Returns
-------

- **PVT**: 
  An instance of the `PVT` class configured as a PVT-Medium vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.pvt_medium()
    >>> print(model)
