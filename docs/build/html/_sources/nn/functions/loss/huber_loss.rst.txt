nn.functional.huber_loss
========================

.. autofunction:: lucid.nn.functional.huber_loss

The `huber_loss` function computes the Huber loss, 
which is less sensitive to outliers than the MSE loss.

Function Signature
------------------

.. code-block:: python

    def huber_loss(
        input_: Tensor,
        target: Tensor,
        delta: float = 1.0,
        reduction: _ReductionType | None = "mean",
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor of shape (N, ...), where N is the batch size.

- **target** (*Tensor*):
    The target tensor of shape matching `input_`.

- **delta** (*float*, optional):
    The threshold at which to change between squared error and absolute error. Default: 1.0.

- **reduction** (*str | None*, optional):
    Specifies the reduction to apply: 'mean', 'sum', or 'none'. Default: 'mean'.

Returns
-------

- **Tensor**:
    The computed Huber loss as a scalar or tensor, depending on the reduction method.

Examples
--------

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([1.0, 2.0, 3.0])
    >>> target = Tensor([1.5, 2.5, 3.5])
    >>> loss = F.huber_loss(input_, target)
    >>> print(loss)
    Tensor(...)  # Scalar loss value