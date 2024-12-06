nn.functional.cross_entropy
===========================

.. autofunction:: lucid.nn.functional.cross_entropy

The `cross_entropy` function computes the cross-entropy 
loss for multi-class classification tasks.

Function Signature
------------------

.. code-block:: python

    def cross_entropy(
        input_: Tensor,
        target: Tensor,
        weight: Tensor | None = None,
        reduction: _ReductionType | None = "mean",
        eps: float = 1e-7,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor of shape (N, C), 
    where N is the batch size and C is the number of classes.

- **target** (*Tensor*):
    The target tensor of shape (N,) containing class indices in the range [0, C-1].

- **weight** (*Tensor | None*, optional):
    A manual rescaling weight tensor of shape (C,). Default: None.

- **reduction** (*str | None*, optional):
    Specifies the reduction to apply: 'mean', 'sum', or 'none'. Default: 'mean'.

- **eps** (*float*, optional):
    A small constant added for numerical stability. Default: 1e-7.

Returns
-------

- **Tensor**:
    The computed cross-entropy loss as a scalar or tensor, 
    depending on the reduction method.

Examples
--------

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[2.0, 1.0, 0.1]])
    >>> target = Tensor([0])
    >>> loss = F.cross_entropy(input_, target)
    >>> print(loss)
    Tensor(...)  # Scalar loss value