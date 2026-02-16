lucid.linalg.eig
================

.. autofunction:: lucid.linalg.eig

The `eig` function computes the eigenvalues and eigenvectors of a square matrix.

Function Signature
------------------

.. code-block:: python

    def eig(a: Tensor) -> tuple[Tensor, Tensor]

Parameters
----------

- **a** (*Tensor*): 
    The input tensor, which must be a square matrix (same number of rows and columns).

Returns
-------

- **Tuple[Tensor, Tensor]**: A tuple containing two tensors:
    - The first tensor contains the eigenvalues of the input matrix.
    - The second tensor contains the eigenvectors of the input matrix, 
      each column corresponding to an eigenvector.

Forward Calculation
-------------------

The forward calculation for `eig` computes the eigenvalues and eigenvectors of a square matrix :math:`\mathbf{A}`. 
The eigenvalues :math:`\lambda` and eigenvectors :math:`\mathbf{v}` satisfy the equation:

.. math::

    \mathbf{A} \mathbf{v} = \lambda \mathbf{v}

where :math:`\mathbf{A}` is the input matrix, :math:`\mathbf{v}` is the eigenvector, 
and :math:`\lambda` is the eigenvalue.

Backward Gradient Calculation
-----------------------------

Let the eigenvalues of the input matrix :math:`\mathbf{A}` be :math:`\lambda_1, \lambda_2, \dots, \lambda_n` 
and the corresponding eigenvectors be :math:`\mathbf{v}_1, \mathbf{v}_2, \dots, \mathbf{v}_n`. 

For a given eigenvalue :math:`\lambda_i` and eigenvector :math:`\mathbf{v}_i`, the gradients of 
the eigenvalue and eigenvector with respect to the input matrix :math:`\mathbf{A}` 
can be computed using the following formulas:

1. **Gradient of the eigenvalue** with respect to the input matrix :math:`\mathbf{A}`:

    .. math::

        \frac{\partial \lambda_i}{\partial \mathbf{A}} = \mathbf{v}_i \mathbf{v}_i^\top

2. **Gradient of the eigenvector** with respect to the input matrix :math:`\mathbf{A}`:

    The gradient of the eigenvector :math:`\mathbf{v}_i` with respect to the input matrix 
    :math:`\mathbf{A}` is more complex and involves a perturbation of the matrix around the eigenvector:

    .. math::

        \frac{\partial \mathbf{v}_i}{\partial \mathbf{A}} = 
        \mathbf{v}_i (\mathbf{v}_i^\top \mathbf{A} - \lambda_i \mathbf{v}_i^\top)

These gradients are propagated through the eigenvectors and eigenvalues during backpropagation.

Raises
------

.. attention::

    - **ValueError**: If the input tensor is not a square matrix.
    
    - **LinAlgError**: If the eigenvalues and eigenvectors cannot be computed 
      (e.g., for non-diagonalizable matrices or matrices with complex eigenvalues).

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[4.0, -2.0], [1.0,  1.0]])
    >>> eigvals, eigvecs = lucid.linalg.eig(a)
    >>> print(eigvals)
    Tensor([3.0, 2.0])
    >>> print(eigvecs)
    Tensor([[ 0.89442719, -0.70710678],
            [ 0.4472136 ,  0.70710678]])

.. note::

    - Eigenvalues and eigenvectors describe the scaling and directions of transformation in a vector space.
    - The input tensor must have two dimensions and be square, i.e., :math:`a.shape[0] == a.shape[1]`.
    - If the matrix is not diagonalizable, or has complex eigenvalues, this function may raise an error.
