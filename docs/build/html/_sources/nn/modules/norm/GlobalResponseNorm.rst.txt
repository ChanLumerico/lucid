nn.GlobalResponseNorm
=====================

.. autoclass:: lucid.nn.GlobalResponseNorm

The `GlobalResponseNorm` class implements global response normalization as a 
neural network module, with learnable scaling and shifting parameters for each channel.

Class Signature
---------------

.. code-block:: python

    class GlobalResponseNorm(nn.Module):
        def __init__(self, channels: int, eps: float = 1e-6) -> None

Parameters
----------

- **channels** (*int*):
  The number of channels in the input tensor. 
  Used to initialize learnable parameters `gamma` and `beta`.

- **eps** (*float*, optional):
  A small constant added to the denominator for numerical stability during normalization. 
  Default: 1e-6.

Forward Calculation
-------------------

.. math::

    y_{n, c, h, w} = \gamma_c \cdot \frac{x_{n, c, h, w}}
    {\sqrt{\frac{1}{H \cdot W} \sum_{h=1}^H \sum_{w=1}^W x_{n, c, h, w}^2 + \epsilon}} + 
    \beta_c

where:

- :math:`x_{n, c, h, w}` is the input value at batch index :math:`n`, 
  channel :math:`c`, height :math:`h`, and width :math:`w`.
- :math:`\gamma_c` and :math:`\beta_c` are the learnable scaling and shifting parameters.
- :math:`H` and :math:`W` are the height and width of the feature map.
- :math:`\epsilon` is a small constant for numerical stability.

Examples
--------

Using the `GlobalResponseNorm` module:

.. code-block:: python

    >>> import lucid.nn as nn
    >>> input_ = Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])  # Shape: (1, 1, 2, 2)
    >>> grn = nn.GlobalResponseNorm(channels=1)
    >>> output = grn(input_)
    >>> print(output)
    Tensor(...)  # Normalized values
