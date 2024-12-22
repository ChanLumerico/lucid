nn.AdaptiveAvgPool3d
====================

.. autoclass:: lucid.nn.AdaptiveAvgPool3d

The `AdaptiveAvgPool3d` performs adaptive average pooling on a 3D input tensor, 
dynamically determining kernel size, stride, and padding to produce a specified output size.

Class Signature
---------------

.. code-block:: python

    class AdaptiveAvgPool3d(nn.Module):
        def __init__(self, output_size: tuple[int, int, int] | int)

Parameters
----------

- **output_size** (*tuple[int, int, int] | int*):
  The desired output size :math:`(D_{out}, H_{out}, W_{out})`. 
  If an integer is provided, the same size is used for all three dimensions.

Attributes
----------

- **output_size** (*tuple[int, int, int] | int*):
  Stores the target output size for the pooling operation.

Forward Calculation
-------------------

The `AdaptiveAvgPool3d` module applies adaptive average pooling on a 3D input tensor \
of shape :math:`(N, C, D, H, W)`, producing an output tensor of shape 
:math:`(N, C, D_{out}, H_{out}, W_{out})`. The kernel size, stride, \
and padding are dynamically determined based on the input size :math:`D, H, W` 
and the desired output size :math:`D_{out}, H_{out}, W_{out}`.

Behavior
--------

The forward calculation for the module follows the same logic as the 
functional API (`adaptive_avg_pool3d`). It dynamically adjusts pooling parameters 
to ensure the target output size is achieved.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.nn as nn

    # Input tensor with shape (1, 3, 16, 16, 16)
    input_ = Tensor([[[[[1.0] * 16] * 16] * 16] * 3])

    # AdaptiveAvgPool3d module with output size (4, 4, 4)
    pool = nn.AdaptiveAvgPool3d(output_size=(4, 4, 4))

    output = pool(input_)
    print(output)  # Shape: (1, 3, 4, 4, 4)

**Output Explanation**

The input tensor is adaptively pooled to produce a tensor with depth, height, 
and width of 4, averaging over evenly spaced regions.

**Advanced Example with Variable Batch Size**

.. code-block:: python

    # Input tensor with batch size 2
    input_ = Tensor([
        [[[[1.0] * 8] * 8] * 8],
        [[[[2.0] * 12] * 12] * 12]
    ])

    pool = nn.AdaptiveAvgPool3d(output_size=(2, 2, 2))

    output = pool(input_)
    print(output)  # Shape: (2, 3, 2, 2, 2)

**Explanation**

The pooling dynamically adjusts for each input's depth, height, and width, 
producing a consistent output size of (2, 2, 2) across all samples in the batch.