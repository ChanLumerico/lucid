nn.functional.adaptive_avg_pool1d
=================================

.. autofunction:: lucid.nn.functional.adaptive_avg_pool1d

The `adaptive_avg_pool1d` performs adaptive average pooling on a 1D input tensor, 
dynamically determining kernel size, stride, and padding to produce a specified output size.

Function Signature
------------------

.. code-block:: python

    def adaptive_avg_pool1d(input_: Tensor, output_size: int) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor of shape :math:`(N, C, L)`, where :math:`N` is the batch size, 
  :math:`C` is the number of channels, and :math:`L` is the input length.

- **output_size** (*int*):
  The desired output length :math:`L_{out}`. The pooling operation dynamically 
  adjusts to achieve this size.

Returns
-------

- **Tensor**:
  The result of adaptive average pooling, with shape :math:`(N, C, L_{out})`.

Behavior
--------

The `adaptive_avg_pool1d` function computes kernel size, stride, 
and padding dynamically based on the input size :math:`L` and the target output size :math:`L_{out}`. The operation averages over the computed kernel regions to produce the output tensor.

Forward Calculation
-------------------

The formula for the output length :math:`L_{out}` is derived as:

.. math::

    L_{out} = \frac{L + 2 \cdot \text{padding} - \text{kernel\_size}}{\text{stride}} + 1

where :math:`\text{padding}` is computed symmetrically to ensure coverage of the input tensor.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.nn.functional as F

    # Input tensor with shape (1, 3, 10)
    input_ = Tensor([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]])

    # Adaptive average pooling to output size 5
    output = F.adaptive_avg_pool1d(input_, output_size=5)

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

    output = F.adaptive_avg_pool1d(input_, output_size=3)

    print(output)  # Shape: (2, 3, 3)

**Explanation**

The pooling dynamically adjusts for each input's length, 
producing a consistent output size of 3 across all samples in the batch.
