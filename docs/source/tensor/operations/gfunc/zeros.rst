lucid.zeros
===========

.. autofunction:: lucid.zeros

The `zeros` function creates a tensor filled with zeros of the specified shape and data type.

Function Signature
------------------

.. code-block:: python

    def zeros(
        shape: _ShapeLike,
        dtype: Any = np.float32,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> Tensor

Parameters
----------

- **shape** (*_ShapeLike*): 
    The shape of the output tensor. Can be a list or tuple of integers.

- **dtype** (*Any*, optional): 
    The data type of the elements in the tensor. Defaults to `np.float32`.

- **requires_grad** (*bool*, optional): 
    If `True`, the resulting tensor will be part of the computation graph and 
    capable of tracking gradients. Defaults to `False`.

- **keep_grad** (*bool*, optional): 
    If `True`, the gradient history will be preserved even if the tensor does not 
    require gradients. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor filled with zeros, having the specified shape and data type.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> z = lucid.zeros((2, 3))
    >>> print(z)
    Tensor([[0. 0. 0.]
            [0. 0. 0.]])

The `zeros` function also allows specifying `requires_grad`:

.. code-block:: python

    >>> z = lucid.zeros((2, 3), requires_grad=True)
    >>> print(z.requires_grad)
    True

.. note::

    - This function is often used for initialization purposes in neural 
      networks and other numerical applications.

    - The `requires_grad` parameter is useful for differentiable operations 
      in computation graphs.

    - If `keep_grad` is set to `True`, the tensor will retain its gradient 
      history even if `requires_grad` is `False`.
