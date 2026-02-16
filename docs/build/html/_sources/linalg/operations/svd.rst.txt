lucid.linalg.svd
================

.. autofunction:: lucid.linalg.svd

The `svd` function computes the Singular Value Decomposition (SVD) 
of a matrix, decomposing it into three matrices: an orthogonal matrix **U**, 
a diagonal matrix **S**, and an orthogonal matrix **V**.

Function Signature
------------------

.. code-block:: python

    def svd(a: Tensor, full_matrices: bool = True) -> tuple[Tensor, Tensor, Tensor]

Parameters
----------

- **a** (*Tensor*): 
    The input tensor, which must be a two-dimensional matrix.

- **full_matrices** (*bool*, optional):
    If `True`, the full-sized **U** and **V** matrices are returned. 
    If `False`, the economy-sized versions are returned. 
    Default is `True`.

Returns
-------

- **Tuple[Tensor, Tensor, Tensor]**: A tuple containing three tensors:
    - **U** (*Tensor*): An orthogonal matrix containing the left singular vectors.
    - **S** (*Tensor*): A diagonal tensor containing the singular values.
    - **V** (*Tensor*): An orthogonal matrix containing the right singular vectors.

Forward Calculation
-------------------

The forward calculation for `svd` decomposes a matrix :math:`\mathbf{A}` 
into the product of three matrices:

.. math::

    \mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^\top

where:

- :math:`\mathbf{U}` is an orthogonal matrix whose columns are the left singular vectors.
- :math:`\mathbf{S}` is a diagonal matrix with non-negative real numbers on the diagonal, representing the singular values.
- :math:`\mathbf{V}` is an orthogonal matrix whose columns are the right singular vectors.

Backward Gradient Calculation
-----------------------------

Given the Singular Value Decomposition :math:`\mathbf{A} = \mathbf{U} \mathbf{S} \mathbf{V}^\top`, 
the gradients of **U**, **S**, and **V** with respect to the input matrix :math:`\mathbf{A}` 
can be computed using the following formulas:

1. **Gradient of the singular values** with respect to the input matrix :math:`\mathbf{A}`:

    .. math::

        \frac{\partial \mathbf{S}}{\partial \mathbf{A}} = 
        \mathbf{U}^\top \frac{\partial \mathbf{A}}{\partial \mathbf{A}} \mathbf{V}

2. **Gradient of the left singular vectors** with respect to the input matrix :math:`\mathbf{A}`:

    The gradient involves projecting the perturbation onto the space orthogonal to **U** and **V**:

    .. math::

        \frac{\partial \mathbf{U}}{\partial \mathbf{A}} = 
        \mathbf{U} \mathbf{S}^{-1} \left( \mathbf{U}^\top \frac{\partial \mathbf{A}}{\partial \mathbf{A}} \mathbf{V} - 
        \mathbf{V}^\top \frac{\partial \mathbf{A}}{\partial \mathbf{A}}^\top \mathbf{U} \right)

3. **Gradient of the right singular vectors** with respect to the input matrix :math:`\mathbf{A}`:

    Similarly, the gradient is computed by projecting onto the orthogonal space:

    .. math::

        \frac{\partial \mathbf{V}}{\partial \mathbf{A}} = 
        \mathbf{V} \mathbf{S}^{-1} \left( \mathbf{V}^\top \frac{\partial \mathbf{A}}{\partial \mathbf{A}}^\top \mathbf{U} - 
        \mathbf{U}^\top \frac{\partial \mathbf{A}}{\partial \mathbf{A}} \mathbf{V} \right)

These gradients are propagated through **U**, **S**, and **V** during backpropagation, 
enabling the optimization of models that incorporate SVD.

Raises
------

.. attention::

    - **ValueError**: If the input tensor is not a two-dimensional matrix.
    
    - **LinAlgError**: If the SVD cannot be computed 
      (e.g., if the input matrix contains NaNs or infinities).

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1.0, 0.0], [0.0, 1.0]])
    >>> u, s, v = lucid.linalg.svd(a)
    >>> print(u)
    Tensor([[1.0, 0.0],
            [0.0, 1.0]])
    >>> print(s)
    Tensor([1.0, 1.0])
    >>> print(v)
    Tensor([[1.0, 0.0],
            [0.0, 1.0]])

.. note::

    - Singular Value Decomposition is useful for dimensionality reduction, noise reduction, and solving least squares problems.
    - The input tensor must have two dimensions, i.e., :math:`a.ndim == 2`.
    - If `full_matrices` is `True`, **U** and **V** are returned as full orthogonal matrices. If `False`, they are economy-sized.
    - The singular values in **S** are returned in descending order.
    - This function does not support batch processing; each input must be a single matrix.