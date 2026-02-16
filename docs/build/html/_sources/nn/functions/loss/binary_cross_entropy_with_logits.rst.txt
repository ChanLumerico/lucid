nn.functional.binary_cross_entropy_with_logits
==============================================

.. autofunction:: lucid.nn.functional.binary_cross_entropy_with_logits

The `binary_cross_entropy_with_logits` function computes the binary 
cross-entropy loss directly from raw logits (before sigmoid activation), 
using a numerically stable formulation.

Function Signature
------------------

.. code-block:: python

    def binary_cross_entropy_with_logits(
        input_: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: str | None = "mean",
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor containing raw logits of shape (N, ...), where N is the batch size.

- **target** (*Tensor*):
  The target tensor of shape matching `input_`, with binary values (0 or 1).

- **weight** (*Tensor | None*, optional):
  A manual rescaling weight tensor of shape matching `input_`. Default: None.

- **reduction** (*str | None*, optional):
  Specifies the reduction to apply: 'mean', 'sum', or 'none'. Default: 'mean'.

Returns
-------

- **Tensor**:
  The computed binary cross-entropy loss as a scalar or tensor, 
  depending on the reduction method.

Examples
--------

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([2.0, -1.0])  # raw logits
    >>> target = Tensor([1.0, 0.0])
    >>> loss = F.binary_cross_entropy_with_logits(input_, target)
    >>> print(loss)
    Tensor(...)  # Scalar loss value
