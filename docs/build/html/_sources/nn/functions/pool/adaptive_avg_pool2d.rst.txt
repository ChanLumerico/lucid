nn.functional.adaptive_avg_pool2d
=================================

.. autofunction:: lucid.nn.functional.adaptive_avg_pool2d

The `adaptive_avg_pool2d` performs adaptive average pooling on a 2D input tensor, 
dynamically determining kernel size, stride, and padding to produce a specified output size.

Function Signature
------------------

.. code-block:: python

    def adaptive_avg_pool2d(input_: Tensor, output_size: tuple[int, int] | int) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor of shape :math:`(N, C, H, W)`, where :math:`N` is the batch size, 
  :math:`C` is the number of channels, :math:`H` and :math:`W` are the height and width of the input.

- **output_size** (*tuple[int, int] | int*):
  The desired output size :math:`(H_{out}, W_{out})`. 
  If an integer is provided, the same size is used for both dimensions.

Returns
-------

- **Tensor**:
  The result of adaptive average pooling, with shape :math:`(N, C, H_{out}, W_{out})`.

Behavior
--------

The `adaptive_avg_pool2d` function computes kernel size, stride, 
and padding dynamically based on the input size :math:`H, W` and the target output size :math:`H_{out}, W_{out}`. The operation averages over the computed kernel regions to produce the output tensor.

Forward Calculation
-------------------

The formula for the output dimensions :math:`H_{out}, W_{out}` is derived as:

.. math::

    H_{out} = \frac{H + 2 \cdot \text{padding} - \text{kernel\_size}}{\text{stride}} + 1

    W_{out} = \frac{W + 2 \cdot \text{padding} - \text{kernel\_size}}{\text{stride}} + 1

where :math:`\text{padding}` is computed symmetrically to ensure coverage of the input tensor.

Examples
--------

**Basic Example**

.. code-block:: python

    import lucid.nn.functional as F

    # Input tensor with shape (1, 3, 32, 32)
    input_ = Tensor([[[[1.0] * 32] * 32] * 3])

    # Adaptive average pooling to output size (8, 8)
    output = F.adaptive_avg_pool2d(input_, output_size=(8, 8))

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

    output = F.adaptive_avg_pool2d(input_, output_size=(4, 4))

    print(output)  # Shape: (2, 3, 4, 4)

**Explanation**

The pooling dynamically adjusts for each input's height and width, 
producing a consistent output size of (4, 4) across all samples in the batch.
