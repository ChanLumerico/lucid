nn.AdaptiveAvgPool1d
====================

.. autoclass:: lucid.nn.AdaptiveAvgPool1d

The `AdaptiveAvgPool1d` performs adaptive average pooling on a 1D input tensor, 
dynamically determining kernel size, stride, and padding to produce a specified output size.

Class Signature
---------------

.. code-block:: python

    class AdaptiveAvgPool1d(nn.Module):
        def __init__(self, output_size: int)

Parameters
----------

- **output_size** (*int*):
  The desired output length :math:`L_{out}`. The pooling operation dynamically 
  adjusts to achieve this size.

Attributes
----------

- **output_size** (*int*):
  Stores the target output length for the pooling operation.

Forward Calculation
-------------------

The `AdaptiveAvgPool1d` module applies adaptive average pooling on a 1D 
input tensor of shape :math:`(N, C, L)`, producing an output tensor of shape 
:math:`(N, C, L_{out})`. The kernel size, stride, and padding are dynamically 
determined based on the input size :math:`L` and the desired output size :math:`L_{out}`.

Behavior
--------

The forward calculation for the module follows the same logic as the 
functional API (`adaptive_avg_pool1d`). It dynamically adjusts pooling parameters to 
ensure the target output size is achieved.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.nn as nn

    # Input tensor with shape (1, 3, 10)
    input_ = Tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]])

    # AdaptiveAvgPool1d module with output size 5
    pool = nn.AdaptiveAvgPool1d(output_size=5)

    output = pool(input_)
    print(output)  # Shape: (1, 3, 5)

**Output Explanation**

The input tensor is adaptively pooled to produce a tensor with 5 evenly 
spaced averaged values per channel.

**Advanced Example with Variable Batch Size**

.. code-block:: python

    # Input tensor with batch size 2 and varying lengths
    input_ = Tensor([
        [[1.0, 2.0, 3.0, 4.0]],
        [[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
    ])

    pool = nn.AdaptiveAvgPool1d(output_size=3)

    output = pool(input_)
    print(output)  # Shape: (2, 3, 3)

**Explanation**

The pooling dynamically adjusts for each input's length, 
producing a consistent output size of 3 across all samples in the batch.
