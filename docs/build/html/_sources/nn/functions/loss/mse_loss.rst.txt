nn.functional.mse_loss
======================

.. autofunction:: lucid.nn.functional.mse_loss

The `mse_loss` function computes the mean squared error (MSE) loss, 
commonly used for regression tasks.

Function Signature
------------------

.. code-block:: python

    def mse_loss(
        input_: Tensor, target: Tensor, reduction: _ReductionType | None = "mean"
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor of shape (N, ...), where N is the batch size.

- **target** (*Tensor*):
    The target tensor of shape matching `input_`.

- **reduction** (*str | None*, optional):
    Specifies the reduction to apply: 'mean', 'sum', or 'none'. Default: 'mean'.

Returns
-------

- **Tensor**:
    The computed MSE loss as a scalar or tensor, depending on the reduction method.

Examples
--------

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([1.0, 2.0, 3.0])
    >>> target = Tensor([1.5, 2.5, 3.5])
    >>> loss = F.mse_loss(input_, target)
    >>> print(loss)
    Tensor(...)  # Scalar loss value