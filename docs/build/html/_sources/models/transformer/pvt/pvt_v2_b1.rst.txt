pvt_v2_b1
=========

.. autofunction:: lucid.models.pvt_v2_b1

The `pvt_v2_b1` function instantiates the PVTv2-B1 model, an enhanced variant
in the Pyramid Vision Transformer V2 family. With increased embedding dimensions and token capacity,
it achieves improved representation power while maintaining a compact architecture.

**Total Parameters**: 14,009,000

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_v2_b1(img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PVT_V2

Parameters
----------

- **img_size** (*int*, optional):  
  The input image size. Default is `224`.

- **num_classes** (*int*, optional):  
  The number of output classes for classification. Default is `1000`.

- **in_channels** (*int*, optional):  
  Number of input channels. Default is `3`.

- **kwargs** (*dict*, optional):  
  Additional parameters for customization, including:

- **embed_dims** (*list[int]*):  
  Embedding dimensions across four stages. Default is `[64, 128, 320, 512]`.

- **depths** (*list[int]*):  
  Number of transformer blocks per stage. Default is `[2, 2, 2, 2]`.

- **num_heads** (*list[int]*):  
  Multi-head self-attention head counts per stage. Default is `[1, 2, 5, 8]`.

- **mlp_ratios** (*list[int]*):  
  MLP expansion ratios per stage. Default is `[8, 8, 4, 4]`.

- **sr_ratios** (*list[int]*):  
  Spatial reduction ratios for attention projection. Default is `[8, 4, 2, 1]`.

- **qkv_bias** (*bool*):  
  Whether to include bias terms in query/key/value. Default is `True`.

- **norm_layer** (*Callable*):  
  Normalization layer used in the model. Default is `nn.LayerNorm(eps=1e-6)`.

Returns
-------
- **PVT_V2**:  
  An instance of the `PVT_V2` class configured as a PVTv2-B1 vision transformer.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.pvt_v2_b1()
    >>> print(model)
    PVT_V2(img_size=224, num_classes=1000, patch_size=7, embed_dims=[64, 128, 320, 512],
           num_heads=[1, 2, 5, 8], depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1], linear=False, ...)

