lucid.round
===========

.. autofunction:: lucid.round

The `round` function rounds each element of the input tensor to the nearest integer 
or to the specified number of decimal places.

Function Signature
------------------

.. code-block:: python

    def round(a: Tensor, *, decimals: int = 0) -> Tensor

Parameters
----------

- **a** (*Tensor*):  
  The input tensor to round.

- **decimals** (*int*, optional):  
  Number of decimal places to round to. Defaults to 0 (round to nearest integer).  
  Can be negative to round to powers of ten.

Returns
-------

- **Tensor**:  
  A new tensor with each element rounded to the specified number of decimals.  
  The output tensor has the same shape and device as the input.

Mathematical Expression
-----------------------

.. math::

    \text{round}(x_i, d) = \mathrm{round}(x_i \cdot 10^d) \cdot 10^{-d}

.. warning::

   This function is **non-differentiable**, so its gradient is always zero.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.Tensor([1.2345, 2.7182, -3.1415])
    >>> lucid.round(x)
    Tensor([1.0, 3.0, -3.0], grad=None)

    >>> lucid.round(x, decimals=2)
    Tensor([1.23, 2.72, -3.14], grad=None)

.. tip::

    Use `round(..., decimals=n)` to perform float-precision rounding, 
    but keep in mind this operation is non-differentiable and mostly used 
    in evaluation or data preprocessing.
