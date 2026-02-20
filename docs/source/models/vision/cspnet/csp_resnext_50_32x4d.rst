csp_resnext_50_32x4d
====================

.. autofunction:: lucid.models.csp_resnext_50_32x4d

The `csp_resnext_50_32x4d` function constructs a CSPNet variant using 
*ResNeXt* blocks as its transformation units. This combines 
the **split-transform-merge** topology of CSPNet with the grouped convolution 
efficiency of ResNeXt, yielding strong accuracy with lower computational 
redundancy.

This configuration mimics the original ResNeXt-50 (32x4d) layout, partitioned 
with Cross Stage Partial (CSP) connections.

**Total Parameters**: 22,509,864

Function Signature
------------------

.. code-block:: python

    @register_model
    def csp_resnext_50_32x4d(
        num_classes: int = 1000, 
        split_ratio: float = 0.5, 
        stem_channels: int = 64, 
        **kwargs
    ) -> CSPNet

Parameters
----------

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is 1000.

- **split_ratio** (*float*, optional):  
  Ratio of input channels routed to the residual stack in each CSP block. Default is 0.5.

- **stem_channels** (*int*, optional):  
  Output channels for the stem convolution. Default is 64.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments passed to `CSPNet`.

Returns
-------

- **CSPNet**:  
  A CSPNet-based model using ResNeXt-50 32x4d block structure.

Examples
--------

.. code-block:: python

    from lucid.models import csp_resnext_50_32x4d

    model = csp_resnext_50_32x4d(num_classes=1000)
    x = lucid.random.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # torch.Size([1, 1000])
