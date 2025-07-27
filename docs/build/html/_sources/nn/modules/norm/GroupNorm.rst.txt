nn.GroupNorm
============

.. autoclass:: lucid.nn.GroupNorm

The `GroupNorm` module applies Group Normalization over a mini-batch of inputs.
It divides the channels into groups and computes mean and variance within each group.

Group Normalization is effective for small batch sizes and performs consistently across 
varied input lengths and shapes.

Class Signature
---------------

.. code-block:: python

    class lucid.nn.GroupNorm(
        num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True
    )

Parameters
----------

- **num_groups** (*int*):  
  Number of groups to divide the channels into. Must evenly divide `num_channels`.

- **num_channels** (*int*):  
  Total number of channels expected in the input. Used to initialize affine parameters 
  if `affine=True`.

- **eps** (*float*, optional):  
  Small constant added to variance for numerical stability. Default is `1e-5`.

- **affine** (*bool*, optional):  
  If `True`, this module includes learnable affine parameters (scale and shift). 
  Default is `True`.

Attributes
----------

- **weight** (*Tensor* or *None*):  
  Learnable affine scale parameter of shape `(num_channels,)`. 
  Only present if `affine=True`.

- **bias** (*Tensor* or *None*):  
  Learnable affine bias parameter of shape `(num_channels,)`. 
  Only present if `affine=True`.

Forward Calculation
-------------------

The `GroupNorm` module performs the following computation per group:

.. math::

    \mu_g &= \frac{1}{m} \sum_{i=1}^m x_i \\
    \sigma_g^2 &= \frac{1}{m} \sum_{i=1}^m (x_i - \mu_g)^2 \\
    \hat{x}_i &= \frac{x_i - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} \\
    y_i &= \gamma \hat{x}_i + \beta

Where:

- :math:`m` is the number of elements in a group.
- :math:`\gamma` and :math:`\beta` are the optional learnable affine parameters.

.. note::

   - When `num_groups == 1`, this becomes LayerNorm across channels.
   - When `num_groups == num_channels`, it behaves like InstanceNorm.

Examples
--------

.. code-block:: python

    >>> import lucid.nn as nn
    >>> norm = nn.GroupNorm(num_groups=2, num_channels=4)
    >>> x = lucid.random.randn(2, 4, 8, 8)
    >>> y = norm(x)
    >>> print(y.shape)
    (2, 4, 8, 8)

.. admonition:: Backpropagation

    .. code-block:: python

        >>> y.sum().backward()
        >>> print(norm.weight.grad)
        [...]
