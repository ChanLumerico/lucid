lucid.full_like
===============

.. autofunction:: lucid.full_like

The `full_like` function returns a new `Tensor` with the same shape as the input, 
filled with a specified scalar value.

Function Signature
------------------

.. code-block:: python

    def full_like(
        a: Tensor | _ArrayLike,
        fill_value: _Scalar,
        dtype: _BuiltinNumeric | Numeric | None = None,
        requires_grad: bool = False,
        keep_grad: bool = False,
        device: _DeviceType | None = None,
    ) -> Tensor

Parameters
----------

- **a** (*Tensor | array-like*):  
  Reference tensor (or array-like) whose shape will be used for the new tensor.

- **fill_value** (*_Scalar*):  
  The constant value to fill the new tensor with.

- **dtype** (*type | Numeric, optional*):  
  Desired data type for the tensor. Can be a built-in numeric type or a `Numeric` type.  
  If `None`, uses the dtype of `a`.

- **requires_grad** (*bool*, optional):  
  If set to `True`, gradients will be tracked for this tensor. Defaults to `False`.

- **keep_grad** (*bool*, optional):  
  Whether to retain the gradient after a backward pass. Defaults to `False`.

- **device** (*Literal["cpu", "gpu"] | None*, optional):  
  Target device for the new tensor. If `None`, uses the device of `a`.

Returns
-------

- **Tensor**:  
  A new tensor filled with `fill_value`, with shape matching `a`.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1, 2], [3, 4]])
    >>> t = lucid.full_like(a, fill_value=9)
    >>> print(t)
    Tensor([[9, 9],
            [9, 9]], grad=None)

.. code-block:: python

    >>> # With GPU and gradient tracking
    >>> t = lucid.full_like(a, 0.0, dtype=lucid.Float32, requires_grad=True, device="gpu")
    >>> print(t.requires_grad)
    True
    >>> print(t.device)
    gpu

Notes
-----

.. tip::

    This function is useful when creating a tensor that mimics the shape of an existing one, 
    but with uniform values.

.. warning::

    If `device` is not specified, `full_like` defaults to the device of the input tensor `a`.
