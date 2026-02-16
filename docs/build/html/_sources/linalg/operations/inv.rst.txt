lucid.linalg.inv
================

.. autofunction:: lucid.linalg.inv

The `inv` function computes the inverse of a square matrix.

Function Signature
------------------

.. code-block:: python

    def inv(a: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): 
    The input tensor, which must be a square matrix (same number of rows and columns).

Returns
-------

- **Tensor**: 
    A tensor representing the inverse of the input matrix.

Forward Calculation
-------------------

The forward calculation for `inv` is based on the mathematical inverse of a matrix:

.. math::

    \mathbf{A}^{-1} \cdot \mathbf{A} = \mathbf{I}

where :math:`\mathbf{A}` is the input matrix, and :math:`\mathbf{I}` is the identity matrix.

Backward Gradient Calculation
-----------------------------

For a tensor :math:`\mathbf{A}` involved in the `inv` operation, 
the gradient of the inverse with respect to the input is computed as:

.. math::

    \frac{\partial \mathbf{A}^{-1}}{\partial \mathbf{A}} = 
    -\mathbf{A}^{-1} \cdot \frac{\partial \mathbf{A}}{\partial \mathbf{A}} \cdot \mathbf{A}^{-1}

This result uses the property of the matrix inverse and chain rule 
to propagate gradients during backpropagation.

Raises
------

.. attention::

    - **ValueError**: If the input tensor is not a square matrix.
    - **LinAlgError**: If the matrix is singular and cannot be inverted.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> inv_a = lucid.linalg.inv(a)
    >>> print(inv_a)
    Tensor([[-2.  1. ]
            [ 1.5 -0.5]])

.. note::

    - The input tensor must have two dimensions and be square, i.e., `a.shape[0] == a.shape[1]`.

    - This function uses efficient numerical methods to compute the 
      inverse but may raise an error if the matrix is poorly conditioned or singular.

.. caution::

   Computing the inverse of a matrix can be numerically unstable for matrices 
   that are close to singular. 
   
   Consider using other techniques such as LU decomposition or 
   pseudo-inverses for better stability.
