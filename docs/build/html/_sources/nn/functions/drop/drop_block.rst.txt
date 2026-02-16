nn.functional.drop_block
========================

.. autofunction:: lucid.nn.functional.drop_block

The `drop_block` function applies structured regularization by dropping 
contiguous regions (blocks) of a tensor. This is particularly useful for 
spatial inputs like feature maps, encouraging the network to utilize more 
distributed features during training.

Function Signature
------------------

.. code-block:: python

    def drop_block(
        input_: Tensor, block_size: int, p: float = 0.1, eps: float = 1e-7
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor to apply DropBlock. Typically of shape :math:`(N, C, H, W)` 
  for spatial feature maps.

- **block_size** (*int*):
  The size of the square block to drop from the input tensor.

- **p** (*float*, optional):
  The probability of dropping a block. Defaults to `0.1`.

- **eps** (*float*, optional):
  A small value added for numerical stability. Defaults to `1e-7`.

Returns
-------

- **Tensor**:
  The tensor with blocks dropped out, having the same shape as the input.

Regularization
--------------

DropBlock regularization zeros out spatially contiguous regions of the input tensor, 
improving model robustness by forcing the network to rely on distributed features. 

Backward Gradient Calculation
-----------------------------

The gradients flow normally through the retained regions, while the dropped regions 
contribute zero gradient.

Examples
--------

**Basic Example:**

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_tensor = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> output = F.drop_block(input_tensor, block_size=1, p=0.5)
    >>> print(output)
    Tensor([[[[1.0, 0.0], [0.0, 4.0]]]])

**Using Larger Blocks:**

.. code-block:: python

    >>> input_tensor = Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # Shape: (1, 1, 3, 3)
    >>> output = F.drop_block(input_tensor, block_size=2, p=0.3)
    >>> print(output)
    Tensor([...])  # Output with dropped blocks
