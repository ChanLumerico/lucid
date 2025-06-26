nn.functional.adaptive_max_pool2d
=================================

.. autofunction:: lucid.nn.functional.adaptive_max_pool2d

Performs adaptive max pooling on a 2D input tensor to produce the specified 
output height and width.

Function Signature
------------------

.. code-block:: python

    def adaptive_max_pool2d(input_: Tensor, output_size: int | tuple[int, int]) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor of shape :math:`(N, C, H, W)`.

- **output_size** (*int or tuple of int*):
  Desired output size :math:`(H_{out}, W_{out})`.

Returns
-------

- **Tensor**:
  Tensor of shape :math:`(N, C, H_{out}, W_{out})`.

Behavior
--------

Kernel size and stride are computed per spatial dimension to reduce 
the input to the given output size, with max pooling over each region.

Examples
--------

.. code-block:: python

    from lucid.nn.functional import adaptive_max_pool2d

    input_ = Tensor.ones((1, 1, 6, 6))
    output = adaptive_max_pool2d(input_, output_size=(3, 3))
    print(output.shape)  # (1, 1, 3, 3)
