lucid.zeros_like
================

.. autofunction:: lucid.zeros_like

The `zeros_like` function creates a tensor of zeros with the same shape and 
optionally the same data type as a given input tensor.

Function Signature
------------------

.. code-block:: python

    def zeros_like(
        a: Tensor | _ArrayLike,
        dtype: Any = None,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> Tensor

Parameters
----------

- **a** (*Tensor | _ArrayLike*): 
    The input tensor or array whose shape will be used to create the zero-filled tensor.

- **dtype** (*Any*, optional): 
    The data type of the elements in the tensor. If `None`, 
    the data type of **a** will be used. Defaults to `None`.

- **requires_grad** (*bool*, optional): 
    If `True`, the resulting tensor will be part of the computation graph and capable of 
    tracking gradients. Defaults to `False`.

- **keep_grad** (*bool*, optional): 
    If `True`, the gradient history will be preserved even if the tensor does not 
    require gradients. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor filled with zeros, having the same shape as the 
    input tensor **a** and optionally the same data type.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2, 3], [4, 5, 6]])
    >>> z = lucid.zeros_like(a)
    >>> print(z)
    Tensor([[0. 0. 0.]
            [0. 0. 0.]])

You can specify a different data type or gradient settings:

.. code-block:: python

    >>> z = lucid.zeros_like(a, dtype=np.float64, requires_grad=True)
    >>> print(z.dtype, z.requires_grad)
    float64 True

.. note::

    - This function is useful for creating zero-initialized tensors that 
      mirror the shape and optionally the type of another tensor or array.

    - If **dtype** is `None`, the data type of **a** is automatically used for the output tensor.

    - The `requires_grad` parameter is useful for tensors involved in 
      gradient-based computation graphs.

    - The `keep_grad` parameter allows preserving gradient history even for 
      tensors that do not require gradients.
