nn.AdaptiveAvgPool2d
====================

.. autoclass:: lucid.nn.AdaptiveAvgPool2d

The `AdaptiveAvgPool2d` performs adaptive average pooling on a 2D input tensor, 
dynamically determining kernel size, stride, and padding to produce a specified output size.

Class Signature
---------------

.. code-block:: python

    class AdaptiveAvgPool2d(nn.Module):
        def __init__(self, output_size: tuple[int, int] | int)

Parameters
----------

- **output_size** (*tuple[int, int] | int*):
  The desired output size :math:`(H_{out}, W_{out})`. If an integer is provided, 
  the same size is used for both dimensions.

Attributes
----------

- **output_size** (*tuple[int, int] | int*):
  Stores the target output size for the pooling operation.

Forward Calculation
-------------------

The `AdaptiveAvgPool2d` module applies adaptive average pooling on a 2D input tensor 
of shape :math:`(N, C, H, W)`, producing an output tensor of shape 
:math:`(N, C, H_{out}, W_{out})`. The kernel size, stride, 
and padding are dynamically determined based on the input size :math:`H, W` and the 
desired output size :math:`H_{out}, W_{out}`.

Behavior
--------

The forward calculation for the module follows the same logic as the 
functional API (`adaptive_avg_pool2d`). It dynamically adjusts pooling parameters 
to ensure the target output size is achieved.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.nn as nn

    # Input tensor with shape (1, 3, 32, 32)
    input_ = Tensor([[[[1.0] * 32] * 32] * 3])

    # AdaptiveAvgPool2d module with output size (8, 8)
    pool = nn.AdaptiveAvgPool2d(output_size=(8, 8))

    output = pool(input_)
    print(output)  # Shape: (1, 3, 8, 8)

**Output Explanation**

The input tensor is adaptively pooled to produce a tensor with height and width of 8, 
averaging over evenly spaced regions.

**Advanced Example with Variable Batch Size**

.. code-block:: python

    # Input tensor with batch size 2
    input_ = Tensor([
        [[[1.0] * 16] * 16],
        [[[2.0] * 24] * 24]
    ])

    pool = nn.AdaptiveAvgPool2d(output_size=(4, 4))

    output = pool(input_)
    print(output)  # Shape: (2, 3, 4, 4)

**Explanation**

The pooling dynamically adjusts for each input's height and width, 
producing a consistent output size of (4, 4) across all samples in the batch.