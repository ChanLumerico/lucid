lucid.full
==========

.. autofunction:: lucid.full

The `full` function creates a `Tensor` of a given shape and fills 
it with a specified scalar value.

Function Signature
------------------

.. code-block:: python

    def full(
        shape: int | _ShapeLike,
        fill_value: _Scalar,
        dtype: _BuiltinNumeric | Numeric | None = None,
        requires_grad: bool = False,
        keep_grad: bool = False,
        device: _DeviceType = "cpu",
    ) -> Tensor

Parameters
----------

- **shape** (*int | tuple[int] | list[int]*):  
  Desired shape of the output tensor.

- **fill_value** (*_Scalar*):  
  The constant value to fill the tensor with.

- **dtype** (*type | Numeric, optional*):  
  Desired data type for the tensor. Can be one of the Python 
  built-in numeric types (e.g. `float`, `int`) or a `Numeric` type. Defaults to `None`.

- **requires_grad** (*bool*, optional):  
  If set to `True`, gradients will be tracked for this tensor. Defaults to `False`.

- **keep_grad** (*bool*, optional):  
  Whether to retain the gradient after a backward pass. Defaults to `False`.

- **device** (*Literal["cpu", "gpu"]*, optional):  
  Device to create the tensor on. `"cpu"` uses NumPy backend, `"gpu"` 
  uses MLX backend. Defaults to `"cpu"`.

Returns
-------

- **Tensor**:  
  A tensor filled with `fill_value`, of specified shape and dtype.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> t = lucid.full((2, 3), fill_value=7)
    >>> print(t)
    Tensor([[7, 7, 7],
            [7, 7, 7]], grad=None)

.. code-block:: python

    >>> # With gradients enabled
    >>> t = lucid.full((2, 2), 3.14, dtype=lucid.Float32, requires_grad=True)
    >>> print(t.requires_grad)
    True

Notes
-----

.. note::

    This function is functionally similar to `lucid.ones` and `lucid.zeros`, 
    but allows specifying any fill value.

.. warning::

    If `dtype` is unspecified and `fill_value` is a float or int, implicit 
    casting rules apply. Use `dtype` explicitly for precise control.
