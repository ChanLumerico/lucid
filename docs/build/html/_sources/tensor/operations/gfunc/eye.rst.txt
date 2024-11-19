lucid.eye
=========

.. autofunction:: lucid.eye

The `eye` function creates a 2D tensor with ones on the diagonal and zeros elsewhere.

Function Signature
------------------

.. code-block:: python

    def eye(
        N: int,
        M: int | None = None,
        k: int = 0,
        dtype: Any = np.float32,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> Tensor

Parameters
----------

- **N** (*int*): 
    The number of rows in the output tensor.

- **M** (*int | None*, optional): 
    The number of columns in the output tensor. If `None`, the tensor will have shape `(N, N)`. 
    Defaults to `None`.

- **k** (*int*, optional): 
    The index of the diagonal. A value of `0` refers to the main diagonal, `k > 0` 
    refers to an offset above the main diagonal, and `k < 0` refers to an offset below it. 
    Defaults to `0`.

- **dtype** (*Any*, optional): 
    The data type of the elements in the tensor. Defaults to `np.float32`.

- **requires_grad** (*bool*, optional): 
    If `True`, the resulting tensor will be part of the computation graph and capable of 
    tracking gradients. Defaults to `False`.

- **keep_grad** (*bool*, optional): 
    If `True`, the gradient history will be preserved even if the tensor does not 
    require gradients. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A 2D tensor with ones on the specified diagonal and zeros elsewhere.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> e = lucid.eye(3)
    >>> print(e)
    Tensor([[1. 0. 0.]
            [0. 1. 0.]
            [0. 0. 1.]])

You can specify the number of columns and the diagonal offset:

.. code-block:: python

    >>> e = lucid.eye(3, 5, k=1)
    >>> print(e)
    Tensor([[0. 1. 0. 0. 0.]
            [0. 0. 1. 0. 0.]
            [0. 0. 0. 1. 0.]])

.. note::

    - The `eye` function is useful for creating identity matrices or related 
      structures for linear algebra and initialization tasks.

    - Supports creating rectangular tensors and customizing the diagonal index.
