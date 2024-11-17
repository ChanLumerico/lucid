lucid.linalg.det
================

.. autofunction:: lucid.linalg.det

The `det` function computes the determinant of a square matrix.

Function Signature
------------------

.. code-block:: python

    def det(a: Tensor) -> Tensor:
        """Compute the determinant of a matrix."""
        return _func.det(a)

Parameters
----------

- **a** (*Tensor*): 
    The input tensor, which must be a square matrix (same number of rows and columns).

Returns
-------

- **Tensor**: 
    A tensor containing the determinant of the input matrix.

Forward Calculation
-------------------

The forward calculation for `det` is:

.. math::

    \text{det}(\mathbf{A})

where :math:`\mathbf{A}` is the input square matrix.

Backward Gradient Calculation
-----------------------------

For a tensor :math:`\mathbf{A}`, the gradient of the determinant 
with respect to the input is given by:

.. math::

    \frac{\partial \text{det}(\mathbf{A})}{\partial \mathbf{A}} = 
    \text{det}(\mathbf{A}) \cdot (\mathbf{A}^{-1})^\top

This involves the determinant and inverse of the matrix :math:`\mathbf{A}`, 
and the gradients are propagated accordingly during backpropagation.

Raises
------

.. attention:: Exceptions

    - **ValueError**: If the input tensor is not a square matrix.
    
    - **LinAlgError**: If the determinant cannot be computed 
      (e.g., for singular or ill-conditioned matrices).

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> det_a = lucid.linalg.det(a)
    >>> print(det_a)
    Tensor(-2.0)

.. note::

    - The determinant is a scalar value that describes certain properties 
      of a matrix, such as whether it is invertible.

    - The input tensor must have two dimensions and be square, 
      i.e., :math:`a.shape[0] == a.shape[1]`.
