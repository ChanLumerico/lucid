nn.DropBlock
============

.. autoclass:: lucid.nn.DropBlock

The `DropBlock` class is a neural network module that applies structured 
regularization to its input by dropping contiguous blocks of activations. 

This is useful in preventing overfitting and encouraging distributed feature 
usage during training.

Class Signature
---------------

.. code-block:: python

    class DropBlock(nn.Module):
        def __init__(self, block_size: int, p: float = 0.1, eps: float = 1e-7) -> None

Parameters
----------

- **block_size** (*int*):
  The size of the square block to drop from the input tensor.

- **p** (*float*, optional):
  The probability of dropping a block. Defaults to `0.1`.

- **eps** (*float*, optional):
  A small value added for numerical stability. Defaults to `1e-7`.

Forward Calculation
-------------------
The `DropBlock` module applies the DropBlock regularization to the input tensor 
during its forward pass. The function:

.. math::
    \text{output} = \text{drop\_block}(\text{input}, \text{block\_size}, p, \text{eps})

Gradients are computed only for the retained regions of the input tensor 
during backpropagation.

Examples
--------

**Basic Usage:**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_tensor = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> dropblock = nn.DropBlock(block_size=1, p=0.5)
    >>> output = dropblock(input_tensor)
    >>> print(output)
    Tensor([[[[1.0, 0.0], [0.0, 4.0]]]])

**Using Larger Blocks:**

.. code-block:: python

    >>> input_tensor = Tensor([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])  # Shape: (1, 1, 3, 3)
    >>> dropblock = nn.DropBlock(block_size=2, p=0.3)
    >>> output = dropblock(input_tensor)
    >>> print(output)
    Tensor([...])  # Output with dropped blocks

.. note::

  - DropBlock is typically applied during training and is often disabled during evaluation.
  - Ensure `block_size` and `p` are chosen to match the input dimensions and desired 
    regularization strength.

.. tip::

  - Combine `DropBlock` with other regularization techniques like dropout or weight 
    decay for better generalization.
  - Use a scheduler to dynamically adjust the `p` parameter during training.
