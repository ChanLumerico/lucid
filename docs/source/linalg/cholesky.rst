lucid.linalg.cholesky
=====================

.. autofunction:: lucid.linalg.cholesky

The `cholesky` function performs the Cholesky decomposition of a symmetric positive-definite matrix.

Function Signature
------------------

.. code-block:: python

    def cholesky(a: Tensor) -> Tensor:
        """Performs Cholesky decomposition."""
        return _func.cholesky(a)

Parameters
----------

- **a** (*Tensor*): 
    A symmetric, positive-definite matrix :math:`\mathbf{A}` to decompose.

Returns
-------

- **Tensor**: 
    A lower triangular matrix :math:`\mathbf{L}` such that:

    .. math::

        \mathbf{A} = \mathbf{L} \mathbf{L}^\top

Forward Calculation
-------------------

The Cholesky decomposition computes a lower triangular matrix :math:`\mathbf{L}` such that:

.. math::

    \mathbf{A} = \mathbf{L} \mathbf{L}^\top

Here, :math:`\mathbf{L}^\top` is the transpose of the matrix :math:`\mathbf{L}`.

Backward Gradient Calculation
-----------------------------

The gradient of the Cholesky decomposition involves differentiating the decomposition itself. 
For a positive-definite matrix :math:`\mathbf{A}`, the gradient with respect to :math:`\mathbf{L}` 
is computed using matrix calculus techniques.

.. math::

    \frac{\partial \mathbf{A}}{\partial \mathbf{L}} = 
    2 \cdot \mathbf{L} \cdot \frac{\partial \mathbf{L}}{\partial \mathbf{L}^\top}

Raises
------

.. attention:: Exceptions

    - **ValueError**: If the input tensor :math:`\mathbf{A}` is not square.

    - **LinAlgError**: If the input tensor :math:`\mathbf{A}` 
      is not symmetric or not positive definite.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[4.0, 2.0], [2.0, 3.0]])
    >>> l = lucid.linalg.cholesky(a)
    >>> print(l)
    Tensor([[2.0, 0.0], [1.0, 1.41421356]])

.. note::

    - The Cholesky decomposition is unique if :math:`\mathbf{A}` is 
      symmetric and positive definite.
    
    - The input matrix must satisfy these conditions for the function to work as expected.
    
    - The resulting lower triangular matrix :math:`\mathbf{L}` 
      can be used to solve linear systems or compute determinants.
