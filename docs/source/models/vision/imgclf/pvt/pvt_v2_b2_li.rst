pvt_v2_b2_li
============

.. autofunction:: lucid.models.pvt_v2_b2_li

The `pvt_v2_b2_li` function instantiates the linear-attention variant of the PVTv2-B2 model.
This version leverages a lightweight attention mechanism, enabling reduced computational
complexity while maintaining strong performance on vision tasks. The "li" suffix stands
for *linear*, referring to the use of **Linear Spatial Reduction Attention** as proposed in the
PVTv2 paper.

Linear spatial reduction attention improves efficiency by:

- Computing attention in a reduced spatial domain.
- Scaling better with input resolution by avoiding quadratic attention cost.
- Maintaining global receptive fields with significantly fewer operations.

This variant is well-suited for deployment scenarios where compute or memory constraints are tight.

**Total Parameters**: 22,553,512

Function Signature
------------------

.. code-block:: python

    @register_model
    def pvt_v2_b2_li(img_size: int = 224, num_classes: int = 1000, in_channels: int = 3, **kwargs) -> PVT_V2

Parameters
----------

- **img_size** (*int*, optional):  
  The input image size. Default is `224`.

- **num_classes** (*int*, optional):  
  The number of output classes for classification. Default is `1000`.

- **in_channels** (*int*, optional):  
  Number of input channels. Default is `3`.

- **kwargs** (*dict*, optional):  
  Additional parameters passed to `pvt_v2_b2`, with `linear=True` fixed.

Returns
-------
- **PVT_V2**:  
  An instance of the `PVT_V2` class configured as a linear-attention PVTv2-B2 model.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> model = models.pvt_v2_b2_li()
    >>> print(model)
    PVT_V2(img_size=224, num_classes=1000, patch_size=7, embed_dims=[64, 128, 320, 512],
           num_heads=[1, 2, 5, 8], depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], linear=True, ...)

