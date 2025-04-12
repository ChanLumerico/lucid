lucid.diag
==========

.. autofunction:: lucid.diag

The `diag` function extracts a diagonal or constructs a diagonal tensor 
from an input tensor or array.

Function Signature
------------------

.. code-block:: python

    def diag(
        v: Tensor | _ArrayLike,
        k: int = 0,
        dtype: Any = np.float32,
        requires_grad: bool = False,
        keep_grad: bool = False,
        device: _DeviceType | None = None,
    ) -> Tensor

Parameters
----------

- **v** (*Tensor | _ArrayLike*): 
    The input tensor or array. If it is a 1D array, a 2D tensor with the 
    specified diagonal is returned. 
    If it is a 2D array, the specified diagonal is extracted.

- **k** (*int*, optional): 
    The index of the diagonal. A value of `0` refers to the main diagonal, `k > 0` 
    refers to a diagonal above the main diagonal, and `k < 0`
    refers to a diagonal below it. Defaults to `0`.

- **dtype** (*Any*, optional): 
    The data type of the output tensor. Defaults to `np.float32`.

- **requires_grad** (*bool*, optional): 
    If `True`, the resulting tensor will be part of the computation graph and 
    capable of tracking gradients. Defaults to `False`.

- **keep_grad** (*bool*, optional): 
    If `True`, the gradient history will be preserved even if the tensor does not
     require gradients. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor representing the extracted diagonal or the constructed diagonal tensor.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> d = lucid.diag(a)
    >>> print(d)
    Tensor([1, 5, 9])

Constructing a diagonal tensor from a 1D array:

.. code-block:: python

    >>> v = Tensor([1, 2, 3])
    >>> d = lucid.diag(v)
    >>> print(d)
    Tensor([[1. 0. 0.]
            [0. 2. 0.]
            [0. 0. 3.]])

.. note::

    - Supports both diagonal extraction and construction depending on 
      the input's dimensionality.

    - Use the `k` parameter to specify diagonals above or below the main diagonal.
