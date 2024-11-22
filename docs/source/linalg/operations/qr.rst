lucid.linalg.qr
===============

.. autofunction:: lucid.linalg.qr

The `qr` function computes the QR decomposition of a matrix, 
decomposing it into an orthogonal matrix **Q** and an upper triangular 
matrix **R**.

Function Signature
------------------

.. code-block:: python

    def qr(a: Tensor) -> tuple[Tensor, Tensor]

Parameters
----------

- **a** (*Tensor*): 
    The input tensor, which must be a two-dimensional matrix.

Returns
-------

- **Tuple[Tensor, Tensor]**: A tuple containing two tensors:
    - **Q** (*Tensor*): An orthogonal matrix where the columns are orthonormal vectors.
    - **R** (*Tensor*): An upper triangular matrix.

Forward Calculation
-------------------

The forward calculation for `qr` decomposes a matrix :math:`\mathbf{A}` 
into the product of an orthogonal matrix :math:`\mathbf{Q}` and an upper 
triangular matrix :math:`\mathbf{R}`:

.. math::

    \mathbf{A} = \mathbf{Q} \mathbf{R}

where:
- :math:`\mathbf{Q}` is an orthogonal matrix (:math:`\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}`),
- :math:`\mathbf{R}` is an upper triangular matrix.

Backward Gradient Calculation
-----------------------------

Given the QR decomposition :math:`\mathbf{A} = \mathbf{Q} \mathbf{R}`, 
the gradients of **Q** and **R** with respect to the input matrix :math:`\mathbf{A}` 
can be computed as follows:

1. **Gradient of Q** with respect to :math:`\mathbf{A}`:

    The gradient is derived based on the orthogonality of **Q**. 
    It involves projecting the gradient onto the space orthogonal to **Q**.

2. **Gradient of R** with respect to :math:`\mathbf{A}`:

    Since **R** is upper triangular, the gradient computation takes into 
    account the structure of **R** to ensure that the gradients respect the 
    upper triangular form.

These gradients are propagated through **Q** and **R** during backpropagation, 
allowing for the optimization of parameters in models that involve QR decomposition.

Raises
------

.. attention::

    - **ValueError**: If the input tensor is not a two-dimensional matrix.
    
    - **LinAlgError**: If the QR decomposition cannot be computed 
      (e.g., if the input matrix contains NaNs or infinities).

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = lucid.Tensor([[1.0, 2.0], [3.0, 4.0]])
    >>> q, r = lucid.linalg.qr(a)
    >>> print(q)
    Tensor([[-0.31622777, -0.9486833 ],
            [-0.9486833 ,  0.31622777]])
    >>> print(r)
    Tensor([[-3.16227766, -4.42718872],
            [ 0.        ,  0.63245553]])

.. note::

    - QR decomposition is useful for solving linear systems, least squares problems, and eigenvalue algorithms.
    - The input tensor must have two dimensions, i.e., :math:`a.ndim == 2`.
    - The matrix **Q** is orthogonal, meaning :math:`\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}`.
    - The matrix **R** is upper triangular, which means all elements below the main diagonal are zero.
    - This function does not support batch processing; each input must be a single matrix.
