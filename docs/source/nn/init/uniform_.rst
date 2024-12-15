nn.init.uniform_
================

.. autofunction:: lucid.nn.init.uniform_

The `uniform_` function fills the input `tensor` with values sampled from a uniform 
distribution :math:`U(a, b)`, where :math:`a` and :math:`b` are the lower and upper 
bounds of the distribution.

Function Signature
------------------

.. code-block:: python

    def uniform_(tensor: Tensor, a: _Scalar = 0, b: _Scalar = 1) -> None

Parameters
----------

- **tensor** (:class:`Tensor`): 
  The tensor to be initialized.

- **a** (`_Scalar`, optional): 
  The lower bound of the uniform distribution. Defaults to 0.

- **b** (`_Scalar`, optional): 
  The upper bound of the uniform distribution. Defaults to 1.

Returns
-------

- **None**: 
  The function modifies the `tensor` in-place with new values sampled from the 
  uniform distribution.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> from lucid.nn.init import uniform_
    >>> tensor = lucid.zeros((3, 3))
    >>> uniform_(tensor, a=-1, b=1)
    >>> print(tensor)
    Tensor([[ 0.423, -0.234,  0.678],
            [-0.123,  0.654, -0.543],
            [ 0.543, -0.345,  0.234]], requires_grad=False)
