nn.functional.batch_norm
========================

.. autofunction:: lucid.nn.functional.batch_norm

The `batch_norm` function performs batch normalization on the input tensor, 
a technique commonly used to improve training of deep neural networks.

Function Signature
------------------

.. code-block:: python

    def batch_norm(
        input_: Tensor,
        running_mean: Tensor,
        running_var: Tensor,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        training: bool = True,
        momentum: float = 0.1,
        eps: float = 1e-5,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor of shape (N, C, ...) where N is the batch size, 
    C is the number of channels, and ... represents any number of additional dimensions.

- **running_mean** (*Tensor*):
    The running mean tensor of shape (C,). Used for normalization in evaluation mode.

- **running_var** (*Tensor*):
    The running variance tensor of shape (C,). Used for normalization in evaluation mode.

- **weight** (*Tensor | None*, optional):
    The learnable scale parameter of shape (C,). Default: None.

- **bias** (*Tensor | None*, optional):
    The learnable shift parameter of shape (C,). Default: None.

- **training** (*bool*, optional):
    If True, the layer uses batch statistics computed from the input. 
    If False, it uses running statistics. Default: True.

- **momentum** (*float*, optional):
    The value used for the running mean and variance computation. Default: 0.1.

- **eps** (*float*, optional):
    A small constant added to the denominator for numerical stability. Default: 1e-5.

Returns
-------

- **Tensor**:
    The normalized tensor of the same shape as `input_`, 
    with normalized values for each feature channel.

Examples
--------

Performing batch normalization during training:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[1.0, 2.0], [3.0, 4.0]]])  # Shape: (1, 2, 2)
    >>> running_mean = Tensor([0.0, 0.0])  # Shape: (2,)
    >>> running_var = Tensor([1.0, 1.0])  # Shape: (2,)
    >>> weight = Tensor([1.0, 1.0])  # Shape: (2,)
    >>> bias = Tensor([0.0, 0.0])  # Shape: (2,)
    >>> out = F.batch_norm(input_, running_mean, running_var, weight, bias, training=True)
    >>> print(out)
    Tensor(...)  # Normalized values

