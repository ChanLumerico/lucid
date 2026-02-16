csp_resnet_50
=============

.. autofunction:: lucid.models.csp_resnet_50

The `csp_resnet_50` function constructs a Cross Stage Partial *ResNet-50* model, 
which integrates the CSPNet topology with the ResNet bottleneck block as the 
core transformation unit. This design significantly reduces duplicate gradient 
flow and enhances computational efficiency while preserving representational 
capacity.

It follows the CSPNet paradigm where the feature maps are **split**, **partially 
transformed**, and then **merged**. This version uses standard ResNet-50 
stage configurations adapted for CSP integration.

**Total Parameters**: 22,463,016

Function Signature
------------------

.. code-block:: python

    @register_model
    def csp_resnet_50(
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
  The fraction of channels sent through the residual transformation path 
  in each CSP stage. Default is 0.5 (even split).

- **stem_channels** (*int*, optional):  
  Number of channels used in the stem (initial) convolution. Default is 64.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments passed to the base `CSPNet` class for 
  further customization.

Returns
-------

- **CSPNet**:  
  An instance of the generalized CSPNet using ResNet-50 style bottlenecks.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import csp_resnet_50

    # Instantiate a CSP-ResNet-50 model
    model = csp_resnet_50(num_classes=1000)

    # Dummy input tensor (1 image, 3 channels, 224x224)
    x = lucid.random.randn(1, 3, 224, 224)

    # Forward pass
    y = model(x)

    print(y.shape)  # Output: torch.Size([1, 1000])
