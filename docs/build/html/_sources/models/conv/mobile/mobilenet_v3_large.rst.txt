mobilenet_v3_large
==================

.. autofunction:: lucid.models.mobilenet_v3_large

The `mobilenet_v3_large` function creates an instance of the **MobileNet-v3-Large** model, 
a lightweight variant of the MobileNetV3 architecture optimized for mobile and embedded 
applications with limited computational resources. 

This model prioritizes efficiency and speed, making it suitable for low-power environments.

**Total Parameters**: 5,481,198

Function Signature
------------------

.. code-block:: python

    @register_model
    def mobilenet_v3_large(num_classes: int = 1000, **kwargs) -> MobileNet_V3

Parameters
----------
- **num_classes** (*int*, optional):  
  Number of output classes for the classification task. Default is 1000, 
  commonly used for ImageNet.

- **kwargs** (*dict*, optional):  
  Additional keyword arguments passed to the `MobileNet_V3` constructor, 
  allowing for further customization.

Returns
-------
- **MobileNet_V3**:  
  An instance of the `MobileNet_V3` model configured with the `Small` variant's architecture.

Bottleneck Configuration
------------------------

The `mobilenet_v3_large` function uses the following bottleneck configurations:

- Each entry in the configuration corresponds to a layer with parameters:
  `[kernel_size, mid_channels, out_channels, use_se, use_hs, stride]`

  where:
    - **kernel_size**: Size of the convolutional kernel.
    - **mid_channels**: Number of channels after expansion in the bottleneck layer.
    - **out_channels**: Number of output channels.
    - **use_se**: Whether to use squeeze-and-excitation.
    - **use_hs**: Whether to use the Hard-Swish activation function.
    - **stride**: Stride of the convolution.

The full configuration:

.. code-block:: python

    cfg = [
        [3, 16, 16, False, False, 1],
        [3, 64, 24, False, False, 2],
        [3, 72, 24, False, False, 1],
        [5, 72, 40, True, False, 2],
        [5, 120, 40, True, False, 1],
        [5, 120, 40, True, False, 1],
        [3, 240, 80, False, True, 2],
        [3, 200, 80, False, True, 1],
        [3, 184, 80, False, True, 1],
        [3, 184, 80, False, True, 1],
        [3, 480, 112, True, True, 1],
        [3, 672, 112, True, True, 1],
        [5, 672, 160, True, True, 2],
        [5, 960, 160, True, True, 1],
        [5, 960, 160, True, True, 1],
    ]

Examples
--------

**Creating a MobileNet-v3-Large model:**

.. code-block:: python

    >>> from lucid.models import mobilenet_v3_large
    >>> model = mobilenet_v3_large(num_classes=1000)
    >>> print(model)

**Forward pass with MobileNet-v3-Large:**

.. code-block:: python

    >>> from lucid.tensor import Tensor
    >>> input_tensor = Tensor([[...]])  # Input tensor with appropriate shape
    >>> output = model(input_tensor)
    >>> print(output)
