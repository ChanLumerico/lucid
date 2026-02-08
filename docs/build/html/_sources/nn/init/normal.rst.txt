nn.init.normal
==============

.. autofunction:: lucid.nn.init.normal

The `normal` function fills the input `tensor` with values sampled from a normal 
distribution :math:`\mathcal{N}(\mu, \sigma^2)`, where :math:`\mu` is the mean and 
:math:`\sigma` is the standard deviation.

Function Signature
------------------

.. code-block:: python

    def normal(tensor: Tensor, mean: _Scalar = 0.0, std: _Scalar = 1.0) -> None

Parameters
----------

- **tensor** (:class:`Tensor`): 
  The tensor to be initialized.

- **mean** (`_Scalar`, optional): 
  The mean of the normal distribution. Defaults to 0.0.

- **std** (`_Scalar`, optional): 
  The standard deviation of the normal distribution. Defaults to 1.0.

Returns
-------

- **None**: 
  The function modifies the `tensor` in-place with new values sampled from 
  the normal distribution.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> from lucid.nn.init import normal
    >>> tensor = lucid.zeros((3, 3))
    >>> normal(tensor, mean=0, std=1)
    >>> print(tensor)
    Tensor([[ 0.423, -0.234,  0.678],
            [-0.123,  0.654, -0.543],
            [ 0.543, -0.345,  0.234]], requires_grad=False)
