nn.functional.adaptive_max_pool1d
=================================

.. autofunction:: lucid.nn.functional.adaptive_max_pool1d

The `adaptive_max_pool1d` performs adaptive max pooling on a 1D input tensor, 
dynamically determining kernel size, stride, and padding to produce a specified output size.

Function Signature
------------------

.. code-block:: python

    def adaptive_max_pool1d(input_: Tensor, output_size: int) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor of shape :math:`(N, C, L)`.

- **output_size** (*int*):
  Desired output length :math:`L_{out}`.

Returns
-------

- **Tensor**:
  Tensor of shape :math:`(N, C, L_{out})` after adaptive max pooling.

Behavior
--------

This function computes the kernel size, stride, and symmetric padding to best match the 
desired output length. Max pooling is applied over each region.

Forward Calculation
-------------------

.. math::

    L_{out} = \frac{L + 2 \cdot \text{padding} - \text{kernel\_size}}{\text{stride}} + 1

Examples
--------

.. code-block:: python

    from lucid.nn.functional import adaptive_max_pool1d

    input_ = Tensor([[[1, 3, 2, 4, 5, 0, 6, 1, 3, 2]]])
    output = adaptive_max_pool1d(input_, output_size=5)
    print(output)  # Shape: (1, 1, 5)
