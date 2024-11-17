Linear Algebra Package
=======================

The `lucid.linalg` package provides a collection of essential linear algebra utilities, 
designed to seamlessly integrate with the `lucid` library's `Tensor` objects. 

These utilities cover a wide range of operations, including matrix computations, 
solvers for linear systems, decomposition methods, and norm calculations.

Features
--------

- Compute matrix properties, such as determinants, traces, and norms.
- Solve linear systems efficiently.
- Perform matrix decompositions.
- Fully compatible with gradient-based computation.

Examples
--------

The following demonstrates typical usage of the `lucid.linalg` package:

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> b = lucid.Tensor([5.0, 6.0])

    # Solve Ax = b
    >>> x = lucid.linalg.solve(a, b)
    >>> print(x)

    # Compute the determinant of a matrix
    >>> det = lucid.linalg.det(a)
    >>> print(det)

.. important::

    - The package is optimized for use in gradient-based optimization tasks.
    - Most functions support batched operations for efficient computation over multiple matrices.