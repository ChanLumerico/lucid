nn.functional.adaptive_max_pool3d
=================================

.. autofunction:: lucid.nn.functional.adaptive_max_pool3d

Applies 3D adaptive max pooling over a volume input to reach a desired depth, 
height, and width.

Function Signature
------------------

.. code-block:: python

    def adaptive_max_pool3d(input_: Tensor, output_size: int | tuple[int, int, int]) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor of shape :math:`(N, C, D, H, W)`.

- **output_size** (*int or tuple of int*):
  Target output shape :math:`(D_{out}, H_{out}, W_{out})`.

Returns
-------

- **Tensor**:
  Tensor of shape :math:`(N, C, D_{out}, H_{out}, W_{out})`.

Behavior
--------

Kernel size and stride are computed in all three spatial dimensions to match the given 
output size, and max pooling is performed over each resulting subvolume.

Examples
--------

.. code-block:: python

    from lucid.nn.functional import adaptive_max_pool3d

    input_ = Tensor.ones((1, 1, 8, 8, 8))
    output = adaptive_max_pool3d(input_, output_size=(2, 2, 2))
    print(output.shape)  # (1, 1, 2, 2, 2)
