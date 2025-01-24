lucid.linspace
==============

.. autofunction:: lucid.linspace

The `linspace` function generates a tensor containing a sequence of evenly spaced 
values between the specified `start` and `stop` values. It is particularly useful 
for creating ranges of values for testing or initialization.

Function Signature
------------------

.. code-block:: python

    def linspace(
        start: _Scalar,
        stop: _Scalar,
        num: int = 50,
        dtype: type = _base_dtype,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> Tensor

Parameters
----------

- **start** (*_Scalar*):
  The starting value of the sequence.

- **stop** (*_Scalar*):
  The ending value of the sequence.

- **num** (*int*, optional):
  The number of values to generate. Default is 50.

- **dtype** (*type*, optional):
  The data type of the resulting tensor. Default is `_base_dtype`.

- **requires_grad** (*bool*, optional):
  If `True`, gradients will be tracked for the generated tensor. Default is `False`.

- **keep_grad** (*bool*, optional):
  If `True`, gradients will not be cleared after each backward pass. Default is `False`.

Returns
-------

- **Tensor**:
  A tensor containing `num` evenly spaced values between `start` and `stop`, inclusive.

Examples
--------

**Basic Usage**

.. code-block:: python

    >>> import lucid
    >>> linspace_tensor = lucid.linspace(0, 1, num=5)
    >>> print(linspace_tensor)
    Tensor([0.0, 0.25, 0.5, 0.75, 1.0], grad=None)

**Custom Data Type**

.. code-block:: python

    >>> linspace_tensor = lucid.linspace(1, 10, num=5, dtype=float)
    >>> print(linspace_tensor)
    Tensor([ 1.   3.25  5.5   7.75 10.  ], grad=None)

**Tracking Gradients**

.. code-block:: python

    >>> linspace_tensor = lucid.linspace(0, 1, num=5, requires_grad=True)
    >>> print(linspace_tensor.requires_grad)
    True

**Using in a Computation**

.. code-block:: python

    >>> start = lucid.Tensor(1.0, requires_grad=True)
    >>> stop = lucid.Tensor(2.0, requires_grad=True)
    >>> linspace_tensor = lucid.linspace(start, stop, num=5)
    >>> print(linspace_tensor)
    Tensor([1.0, 1.25, 1.5, 1.75, 2.0], grad=None)

.. note::

    - If `num` is less than or equal to 1, a single value equal to `start` is returned.
    - Ensure that `start`, `stop`, and `num` are compatible types to avoid runtime errors.
    - The `linspace` function is particularly useful for interpolation, initialization, 
      or sampling in various deep learning tasks.
