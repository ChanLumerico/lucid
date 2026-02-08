csp_darknet_53
==============

.. autofunction:: lucid.models.csp_darknet_53

The `csp_darknet_53` function builds a variant of CSPNet based on 
DarkNet-53 architecture as used in YOLO-v4. It replaces each standard 
residual stage with a *CSPDarkBlock*, utilizing DarkNet bottlenecks and 
cross-stage partial connections to significantly reduce memory cost and 
duplicate gradient paths.

The model is constructed with 5 CSP stages, closely aligned with the 
original DarkNet-53 design, while maintaining CSP semantics throughout.

**Total Parameters**: 27,278,536

Function Signature
------------------

.. code-block:: python

    @register_model
    def csp_darknet_53(
        num_classes: int = 1000, 
        split_ratio: float = 0.5, 
        stem_channels: int = 32, 
        **kwargs
    ) -> CSPNet

Parameters
----------

- **num_classes** (*int*, optional):  
  Number of output classes for classification. Default is 1000.

- **split_ratio** (*float*, optional):  
  Ratio of channels going through the residual transform branch in each stage. Default is 0.5.

- **stem_channels** (*int*, optional):  
  Channels used in the stem convolution layer. Default is 32.

- **kwargs** (*dict*, optional):  
  Additional customization arguments passed to `CSPNet`.

Returns
-------

- **CSPNet**:  
  A CSPNet-based model using DarkNet-53 backbone structure with CSP wrapping.

Examples
--------

.. code-block:: python

    from lucid.models import csp_darknet_53

    model = csp_darknet_53(num_classes=1000)
    x = lucid.random.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)  # torch.Size([1, 1000])
