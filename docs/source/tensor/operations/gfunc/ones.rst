lucid.ones
==========

.. autofunction:: lucid.ones

The `ones` function creates a tensor filled with ones of the specified shape and data type.

Function Signature
------------------

.. code-block:: python

    def ones(
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
    If `True`, the gradient history will be preserved even if the tensor does 
    not require gradients. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor filled with ones, having the specified shape and data type.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> o = lucid.ones((2, 3))
    >>> print(o)
    Tensor([[1. 1. 1.]
            [1. 1. 1.]])

.. note::

    - This function is commonly used for initialization purposes in 
      neural networks or other numerical computations.
      
    - Gradient-related parameters allow flexibility in gradient tracking 
      for differentiable operations.
