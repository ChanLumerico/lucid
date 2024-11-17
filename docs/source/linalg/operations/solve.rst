lucid.linalg.solve
==================

.. autofunction:: lucid.linalg.solve

The `solve` function solves a linear system of equations 
:math:`\mathbf{A} \mathbf{x} = \mathbf{b}`.

Function Signature
------------------

.. code-block:: python

    def solve(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): 
    The coefficient matrix :math:`\mathbf{A}`. 
    It must be square (same number of rows and columns).

- **b** (*Tensor*): 
    The right-hand side matrix or vector :math:`\mathbf{b}`. 
    Its number of rows must match the number of rows in :math:`\mathbf{A}`.

Returns
-------

- **Tensor**: 
    A tensor containing the solution vector or matrix :math:`\mathbf{x}` 
    to the equation :math:`\mathbf{A} \mathbf{x} = \mathbf{b}`.

Forward Calculation
-------------------

The forward calculation for `solve` finds the solution :math:`\mathbf{x}` that satisfies:

.. math::

    \mathbf{A} \mathbf{x} = \mathbf{b}

Backward Gradient Calculation
-----------------------------

For a linear system, the gradient calculations are based on perturbing the matrices 
:math:`\mathbf{A}` and :math:`\mathbf{b}`. 

If the solution is :math:`\mathbf{x}`, the gradients are computed as follows:

.. math::

    \frac{\partial \mathbf{x}}{\partial \mathbf{A}} = 
    -\mathbf{A}^{-1} \cdot \mathbf{x} \cdot \frac{\partial \mathbf{b}}{\partial \mathbf{A}}

.. math::

    \frac{\partial \mathbf{x}}{\partial \mathbf{b}} = \mathbf{A}^{-1}

This leverages matrix differentiation and propagates gradients efficiently.

Raises
------

.. attention::

    - **ValueError**: If the matrix :math:`\mathbf{A}` 
      is not square or if the dimensions of :math:`\mathbf{b}` do not align with :math:`\mathbf{A}`.

    - **LinAlgError**: If the matrix :math:`\mathbf{A}` 
      is singular and the system cannot be solved.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[3.0, 1.0], [1.0, 2.0]])
    >>> b = lucid.Tensor([9.0, 8.0])
    >>> x = lucid.linalg.solve(a, b)
    >>> print(x)
    Tensor([2.0, 3.0])

.. note::

    - The input tensor :math:`\mathbf{A}` must be invertible 
      to ensure a unique solution exists.

    - If :math:`\mathbf{b}` is a matrix, the function computes 
      solutions for multiple right-hand sides.
