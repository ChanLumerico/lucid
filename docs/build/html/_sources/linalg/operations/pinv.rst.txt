lucid.linalg.pinv
=================

.. autofunction:: lucid.linalg.pinv

The `pinv` function computes the Moore-Penrose pseudo-inverse of a matrix. 
Given any matrix **A**, it returns the pseudo-inverse **A⁺**, 
which generalizes the concept of the inverse to non-square or singular matrices.

Function Signature
------------------

.. code-block:: python

    def pinv(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*):
  The input tensor, which can be any two-dimensional tensor (matrix), 
  regardless of shape or rank.

Returns
-------

- **Tensor**: The pseudo-inverse of the input matrix **a**.

Forward Calculation
-------------------

The forward calculation for `pinv` computes the Moore-Penrose pseudo-inverse 
using Singular Value Decomposition (SVD):

1. **Compute the SVD of A**:

   .. math::

       \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^\mathrm{T}

   where:

   - :math:`\mathbf{U}` is an orthogonal matrix containing the left singular vectors.
   - :math:`\mathbf{\Sigma}` is a diagonal matrix containing the singular values (non-negative real numbers).
   - :math:`\mathbf{V}` is an orthogonal matrix containing the right singular vectors.

2. **Compute the reciprocal of non-zero singular values**:

   .. math::

       \mathbf{\Sigma}^+ = \operatorname{diag}\left( \frac{1}{\sigma_i} \right)

   for each non-zero singular value σᵢ. Singular values close to zero can be 
   regularized to avoid numerical instability.

3. **Compute the pseudo-inverse**:

   .. math::

       \mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^\mathrm{T}

Backward Gradient Calculation
-----------------------------

The gradient of the pseudo-inverse with respect to the input matrix **A** 
is computed using advanced matrix calculus involving the SVD components. 

The derivative is given by:

.. math::

    \frac{\partial \mathbf{A}^+}{\partial \mathbf{A}} = 
    -\mathbf{A}^+ \otimes \mathbf{A}^+ + \mathbf{A}^+ 
    \left( \mathbf{A}^{+\mathrm{T}} \otimes \left( \mathbf{I} - \mathbf{A} \mathbf{A}^+ \right) + 
    \left( \mathbf{I} - \mathbf{A}^+ \mathbf{A} \right) \otimes \mathbf{A}^{+\mathrm{T}} \right) \mathbf{A}^+

In practical implementation, this involves:

1. **Compute the SVD of A** (from the forward pass).

2. **Compute the necessary intermediate matrices**:

   - :math:`\mathbf{S}^{-2}`: Diagonal matrix with elements :math:`1/\sigma_i^2`.
   - **Orthogonal projectors**:

     .. math::

         \mathbf{P}_U = \mathbf{U} \mathbf{U}^\mathrm{T}, \quad \mathbf{P}_V = 
         \mathbf{V} \mathbf{V}^\mathrm{T}

3. **Compute the gradient**:

   .. math::

       \frac{\partial L}{\partial \mathbf{A}} = 
       -\mathbf{A}^+ \frac{\partial L}{\partial \mathbf{A}^+} \mathbf{A}^+ + 
       \mathbf{P}_V \frac{\partial L}{\partial \mathbf{A}^+}^\mathrm{T} 
       \left( \mathbf{I} - \mathbf{A} \mathbf{A}^+ \right ) + 
       \left( \mathbf{I} - \mathbf{A}^+ \mathbf{A} \right ) 
       \frac{\partial L}{\partial \mathbf{A}^+}^\mathrm{T} \mathbf{P}_U

   where :math:`\frac{\partial L}{\partial \mathbf{A}^+}` is the gradient of 
   the loss function with respect to the pseudo-inverse **A⁺**.

**Explanation**:

- The gradient involves several terms accounting for the non-square and potentially singular nature of **A**.
- It ensures that during backpropagation, the gradients are correctly propagated through the pseudo-inverse operation.

Raises
------

.. attention::

  - **LinAlgError**: If the SVD computation does not converge during the calculation 
    of the pseudo-inverse or its gradient.
  - **ValueError**: If the input tensor **a** is not a two-dimensional tensor.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> result = lucid.linalg.pinv(a)
    >>> print(result)
    Tensor([[-1.33333333, -0.33333333,  0.66666667],
            [ 1.08333333,  0.33333333, -0.41666667]])

.. note::

  - The pseudo-inverse is particularly useful for solving least squares problems and 
    systems of linear equations that do not have a unique solution.

  - The function supports backpropagation, allowing it to be used in optimization 
    problems involving pseudo-inverses.

  - Numerical stability is maintained by handling singular values close to zero 
    appropriately during the computation.

Additional Details
------------------

- **Singular Values Near Zero**:

  - Small singular values can cause numerical instability due to division by very small numbers.
  - In practice, a threshold or regularization parameter may be used to avoid dividing by zero or 
    extremely small values.

- **Applications**:

  - The pseudo-inverse is widely used in machine learning, statistics, and engineering for solving 
    ill-posed problems.

  - It is essential in computing solutions to linear systems that are underdetermined or overdetermined.

- **Performance Considerations**:

  - Computing the SVD can be computationally intensive for large matrices.
  - For performance-critical applications, consider using approximations or specialized 
    algorithms optimized for large-scale computations.
