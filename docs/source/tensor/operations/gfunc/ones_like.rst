lucid.ones_like
===============

.. autofunction:: lucid.ones_like

The `ones_like` function creates a tensor of ones with the same 
shape and optionally the same data type as a given input tensor.

Function Signature
------------------

.. code-block:: python

    def ones_like(
        a: Tensor | _ArrayLike,
        dtype: Any = None,
        requires_grad: bool = False,
        keep_grad: bool = False,
        device: _DeviceType | None = None,
    ) -> Tensor

Parameters
----------

- **a** (*Tensor | _ArrayLike*): 
    The input tensor or array whose shape will be used to create 
    the ones-filled tensor.

- **dtype** (*Any*, optional): 
    The data type of the elements in the tensor. 
    If `None`, the data type of **a** will be used. Defaults to `None`.

- **requires_grad** (*bool*, optional): 
    If `True`, the resulting tensor will be part of the computation graph and 
    capable of tracking gradients. Defaults to `False`.

- **keep_grad** (*bool*, optional): 
    If `True`, the gradient history will be preserved even if the tensor does 
    not require gradients. Defaults to `False`.

Returns
-------

- **Tensor**: 
    A tensor filled with ones, having the same shape as the input tensor **a** 
    and optionally the same data type.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2, 3], [4, 5, 6]])
    >>> o = lucid.ones_like(a)
    >>> print(o)
    Tensor([[1. 1. 1.]
            [1. 1. 1.]])

.. note::

    - This function is useful for creating one-initialized tensors mirroring 
      the shape of another tensor or array.

    - The `requires_grad` and `keep_grad` parameters provide flexibility in 
      gradient tracking.
