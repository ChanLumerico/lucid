nn.functional.group_norm
========================

.. autofunction:: lucid.nn.functional.group_norm

The `group_norm` function applies Group Normalization over a mini-batch of inputs.
Unlike BatchNorm or InstanceNorm, GroupNorm divides the channels into groups and 
computes within each group.

Function Signature
------------------

.. code-block:: python

    def group_norm(
        input_: Tensor,
        num_groups: int,
        weight: Tensor | None = None,
        bias: Tensor | None = None,
        eps: float = 1e-5,
    ) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  Input tensor of shape :math:`(N, C, *)`, where :math:`*` can be any number 
  of additional dimensions.

- **num_groups** (*int*):  
  Number of groups to divide the channels into. Must divide the number 
  of channels evenly.

- **weight** (*Tensor*, optional):  
  Learnable affine scale tensor of shape :math:`(C,)`. If provided, 
  it is applied after normalization.

- **bias** (*Tensor*, optional):  
  Learnable affine bias tensor of shape :math:`(C,)`. If provided, 
  it is added after scaling.

- **eps** (*float*, optional):  
  A small value added to the denominator for numerical stability. 
  Default is `1e-5`.

Forward Computation
-------------------

GroupNorm computes the mean and variance across each group of channels and 
normalizes accordingly:

.. math::

    \mu_g &= \frac{1}{m} \sum_{i=1}^m x_i \\
    \sigma_g^2 &= \frac{1}{m} \sum_{i=1}^m (x_i - \mu_g)^2 \\
    y &= \frac{x - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}

where :math:`m = \frac{C}{G} \times \text{prod}(*shape[2:])` and :math:`G` 
is the number of groups.

.. tip::

   If `num_groups == C`, this becomes Instance Normalization.  
   If `num_groups == 1`, it becomes Layer Normalization over channels.

Returns
-------

- **Tensor**:  
    The normalized tensor with the same shape as the input.

Examples
--------

.. code-block:: python

    >>> import lucid.nn.functional as F
    >>> x = Tensor.randn(2, 6, 4, 4, requires_grad=True)
    >>> out = F.group_norm(x, num_groups=3)
    >>> print(out.shape)
    (2, 6, 4, 4)

.. code-block:: python

    >>> # With affine parameters
    >>> weight = Tensor.ones(6, requires_grad=True)
    >>> bias = Tensor.zeros(6, requires_grad=True)
    >>> out = F.group_norm(x, num_groups=3, weight=weight, bias=bias)
    >>> out.backward()

.. warning::

    The number of channels (C) must be divisible by `num_groups`.  
    Otherwise, this function will raise a `ValueError`.

