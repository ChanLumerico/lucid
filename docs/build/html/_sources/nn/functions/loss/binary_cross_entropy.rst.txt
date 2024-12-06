nn.functional.binary_cross_entropy
==================================

.. autofunction:: lucid.nn.functional.binary_cross_entropy

The `binary_cross_entropy` function computes the binary 
cross-entropy loss for binary classification tasks.

Function Signature
------------------

.. code-block:: python

    def binary_cross_entropy(
        input_: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean",
        eps: float = 1e-7,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor containing probabilities of shape (N, ...), where N is the batch size.

- **target** (*Tensor*):
    The target tensor of shape matching `input_` with binary values (0 or 1).

- **weight** (*Tensor | None*, optional):
    A manual rescaling weight tensor of shape matching `input_`. Default: None.

- **reduction** (*str | None*, optional):
    Specifies the reduction to apply: 'mean', 'sum', or 'none'. Default: 'mean'.

- **eps** (*float*, optional):
    A small constant added for numerical stability. Default: 1e-7.

Returns
-------

- **Tensor**:
    The computed binary cross-entropy loss as a scalar or tensor, 
    depending on the reduction method.

Examples
--------

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([0.8, 0.2])
    >>> target = Tensor([1.0, 0.0])
    >>> loss = F.binary_cross_entropy(input_, target)
    >>> print(loss)
    Tensor(...)  # Scalar loss value