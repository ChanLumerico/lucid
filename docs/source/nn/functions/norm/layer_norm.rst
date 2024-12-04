nn.functional.layer_norm
========================

.. autofunction:: lucid.nn.functional.layer_norm

The `layer_norm` function performs layer normalization on the input tensor, 
normalizing across the last dimensions as specified by the normalized shape.

Function Signature
------------------

.. code-block:: python

    def layer_norm(
        input_: Tensor,
        normalized_shape: _ShapeLike,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        eps: float = 1e-5,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
    The input tensor of shape (N, ...), where N is the batch size and ... 
    represents the remaining dimensions.

- **normalized_shape** (*_ShapeLike*):
    The shape of the dimensions to normalize. Must match the trailing 
    dimensions of the input tensor.

- **weight** (*Tensor | None*, optional):
    The learnable scale parameter with shape matching `normalized_shape`. Default: None.

- **bias** (*Tensor | None*, optional):
    The learnable shift parameter with shape matching `normalized_shape`. Default: None.

- **eps** (*float*, optional):
    A small constant added to the denominator for numerical stability. Default: 1e-5.

Returns
-------

- **Tensor**:
    The normalized tensor of the same shape as `input_` 
    with the specified dimensions normalized.

Examples
--------

Performing layer normalization:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[1.0, 2.0], [3.0, 4.0]])  # Shape: (2, 2)
    >>> normalized_shape = (2,)
    >>> weight = Tensor([1.0, 1.0])  # Shape: (2,)
    >>> bias = Tensor([0.0, 0.0])  # Shape: (2,)
    >>> out = F.layer_norm(input_, normalized_shape, weight, bias)
    >>> print(out)
    Tensor(...)  # Normalized values
