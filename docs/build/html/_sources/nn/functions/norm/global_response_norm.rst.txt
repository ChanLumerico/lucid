nn.functional.global_response_norm
==================================

.. autofunction:: lucid.nn.functional.global_response_norm

The `global_response_norm` function applies global response normalization to the 
input tensor, normalizing the values across spatial dimensions and scaling/shifting 
with learnable parameters.

Function Signature
------------------

.. code-block:: python

    def global_response_norm(
        input_: Tensor,
        gamma: Tensor,
        beta: Tensor,
        eps: float = 1e-6,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):
  The input tensor of shape (N, C, H, W) where:
  - N: Batch size
  - C: Number of channels
  - H: Height of the feature map
  - W: Width of the feature map

- **gamma** (*Tensor*):
  The learnable scaling parameter of shape (C,).

- **beta** (*Tensor*):
  The learnable shift parameter of shape (C,).

- **eps** (*float*, optional):
  A small constant added to the denominator for numerical stability. Default: 1e-6.

Returns
-------

- **Tensor**:
  The normalized tensor of the same shape as `input_`, 
  with values scaled and shifted using `gamma` and `beta`.

Mathematical Expression
-----------------------

The global response normalization is computed as:

.. math::

    y_{n, c, h, w} = \gamma_c \cdot \frac{x_{n, c, h, w}}
    {\sqrt{\frac{1}{H \cdot W} \sum_{h=1}^H \sum_{w=1}^W x_{n, c, h, w}^2 + \epsilon}} + 
    \beta_c

where:

- :math:`x_{n, c, h, w}` is the input value at batch index :math:`n`, channel :math:`c`, 
  height :math:`h`, and width :math:`w`.
- :math:`\gamma_c` and :math:`\beta_c` are the scaling and shifting parameters for 
  channel :math:`c`.
- :math:`H` and :math:`W` are the height and width of the feature map.
- :math:`\epsilon` is a small constant for numerical stability.

Examples
--------

Performing global response normalization:

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> gamma = Tensor([1.0])  # Shape: (1,)
    >>> beta = Tensor([0.0])   # Shape: (1,)
    >>> out = F.global_response_norm(input_, gamma, beta)
    >>> print(out)
    Tensor(...)  # Normalized values
