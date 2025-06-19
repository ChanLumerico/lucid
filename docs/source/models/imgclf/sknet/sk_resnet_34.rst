sk_resnet_34
============

.. autofunction:: lucid.models.sk_resnet_34

The `sk_resnet_34` function constructs a Selective Kernel (SK) variant of the ResNet-34 architecture. 
This model utilizes the `SKNet` class and `_SKResNetModule` as its building block, enabling 
adaptive multi-scale feature fusion through selective kernels.

**Total Parameters**: 45,895,512

Function Signature
------------------

.. code-block:: python

    @register_model
    def sk_resnet_34(num_classes: int = 1000, **kwargs) -> SKNet

Parameters
----------
- **num_classes** (*int*, optional):
  Number of output classes for the final fully connected layer. Default: 1000.

- **kwargs** (*dict*, optional):
  Additional keyword arguments passed to the `SKNet` class.

Returns
-------
- **SKNet**:
  A Selective Kernel ResNet-34 model instance.

Description
-----------
The `sk_resnet_34` function initializes a ResNet-34-like architecture that uses SK blocks for 
adaptive multi-scale feature fusion. The `layers` parameter defines the configuration for the 
number of blocks in each stage:

.. math::

    \text{layers} = [3, 4, 6, 3]

These layers correspond to the stages of the ResNet-34 backbone, with the specified number of 
blocks per stage.

Examples
--------

**Basic Example**:

.. code-block:: python

    >>> from lucid.models import sk_resnet_34
    >>> model = sk_resnet_34(num_classes=1000, kernel_sizes=[3, 5])
    >>> input_tensor = Tensor(np.random.randn(8, 3, 224, 224))  # Shape: (N, C, H, W)
    >>> output = model(input_tensor)  # Forward pass
    >>> print(output.shape)
    (8, 1000)
