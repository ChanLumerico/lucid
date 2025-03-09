pvt_tiny
========

.. autofunction:: lucid.models.pvt_tiny

The `pvt_tiny` function instantiates the PVT-Tiny model, a lightweight variant of the 
Pyramid Vision Transformer (PVT). PVT-Tiny utilizes a hierarchical transformer 
architecture with progressive spatial reduction, enabling efficient global feature 
learning while maintaining computational efficiency.

**Total Parameters**: 12,457,192

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_tiny(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

Parameters
----------

- **img_size** (*int*, optional):  
  The input image size. Default is `224`.

- **num_classes** (*int*, optional):  
  The number of output classes for classification. Default is `1000`.

- **kwargs** (*dict*, optional):  
  Additional parameters for customization, including:

  - **embed_dims** (*list[int]*):  
    A list specifying the embedding dimensions for different stages.  
    The default for PVT-Tiny is `[64, 128, 320, 512]`, indicating the number of 
    channels at each stage of the model.

  - **depths** (*list[int]*):  
    A list specifying the number of transformer blocks in each stage.  
    The default configuration for PVT-Tiny is `[2, 2, 2, 2]`, representing
    the depth of the transformer at each stage.

  - **num_heads** (*list[int]*):  
    A list specifying the number of attention heads in each stage.  
    The default values for PVT-Tiny are `[1, 2, 5, 8]`, enabling multi-head 
    self-attention at different scales.

  - **mlp_ratios** (*list[float]*):  
    The expansion ratio for the MLP layers in each stage.  
    Default is `[8.0, 8.0, 4.0, 4.0]`.

  - **sr_ratios** (*list[float]*):  
    A list specifying the spatial reduction ratio for key and value projections 
    in each stage.  
    Default is `[8.0, 4.0, 2.0, 1.0]`.

Returns
-------
- **PVT**:  
  An instance of the `PVT` class configured as a PVT-Tiny vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.pvt_tiny()
    >>> print(model)
    PVT(img_size=224, num_classes=1000, embed_dims=[64, 128, 320, 512], 
        depths=[2, 2, 2, 2], num_heads=[1, 2, 5, 8], sr_ratios=[8.0, 4.0, 2.0, 1.0], ...)
