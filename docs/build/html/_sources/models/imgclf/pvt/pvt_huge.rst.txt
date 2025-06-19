pvt_huge
========

.. autofunction:: lucid.models.pvt_huge

The `pvt_huge` function instantiates the PVT-Huge model, the largest and most powerful 
variant of the Pyramid Vision Transformer (PVT). This variant is designed for large-scale 
and highly complex vision tasks, providing the most detailed hierarchical feature 
representation. 

The PVT-Huge model provides excellent performance on challenging tasks and is ideal 
for large-scale datasets.

**Total Parameters**: 286,706,920

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_huge(img_size: int = 224, num_classes: int = 1000, **kwargs) -> PVT

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
    The default configuration for PVT-Huge is `[128, 256, 512, 768]`. 
    This allows for the finest resolution of feature extraction at each stage, 
    making it suitable for large-scale datasets and complex tasks.

  - **depths** (*list[int]*): 
    A list specifying the number of transformer blocks in each stage. 
    The default values for PVT-Huge are `[3, 10, 60, 3]`. This provides very deep 
    models in the intermediate layers for increased abstraction and representation learning.

  - **num_heads** (*list[int]*): 
    The number of self-attention heads in each transformer stage. 
    The default values are `[2, 4, 8, 12]`. The number of heads increases 
    progressively to capture a richer set of dependencies across the hierarchical layers.

  - **mlp_ratios** (*list[float]*): 
    The ratio of MLP hidden dimensions to the embedding dimension in each stage. 
    Default values are `[8.0, 8.0, 4.0, 4.0]`, balancing computational efficiency 
    with expressiveness.

  - **sr_ratios** (*list[float]*): 
    The spatial reduction ratios used in the self-attention mechanism. 
    Default values are `[8.0, 4.0, 2.0, 1.0]`. These values control how much 
    spatial information is preserved across different transformer blocks, 
    optimizing both local and global contextual relationships.

Returns
-------

- **PVT**: 
  An instance of the `PVT` class configured as a PVT-Huge vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.pvt_huge()
    >>> print(model)
