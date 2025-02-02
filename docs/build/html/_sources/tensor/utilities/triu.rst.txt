lucid.triu
==========

.. autofunction:: lucid.triu

The `triu` function extracts the upper triangular part of a tensor,  
setting all elements below a specified diagonal to zero.

Function Signature
------------------

.. code-block:: python

    def triu(input_: Tensor, diagonal: int = 0) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor, which must be at least 2-dimensional.  

- **diagonal** (*int, optional*):  
  Specifies which diagonal to consider (default: `0`).  
  - `0`: Keeps the main diagonal.  
  - `>0`: Excludes more lower rows.  
  - `<0`: Includes elements below the main diagonal.

Returns
-------

- **Tensor**:  
  A new tensor with the same shape as the input,  
  but with all elements below the specified diagonal set to zero.

Forward Calculation
-------------------

The `triu` operation applies an upper-triangular mask to the input tensor:

.. math::

    \mathbf{out}_{ij} =
    \begin{cases}
    \mathbf{X}_{ij}, & \text{if } i - j \leq \text{diagonal} \\
    0, & \text{otherwise}
    \end{cases}

Backward Gradient Calculation
-----------------------------

The gradient for the `triu` operation ensures that only the upper triangular  
portion receives gradients, setting the lower triangular part to zero:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{X}} =
    \mathbf{M}, \quad \text{where} \quad
    M_{ij} =
    \begin{cases}
    1, & \text{if } i - j \leq \text{diagonal} \\
    0, & \text{otherwise}
    \end{cases}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0, 3.0],
    ...             [4.0, 5.0, 6.0],
    ...             [7.0, 8.0, 9.0]], requires_grad=True)
    >>> upper_triangular = lucid.triu(a)
    >>> print(upper_triangular)
    Tensor([[1. 2. 3.]
            [0. 5. 6.]
            [0. 0. 9.]], grad=None)

    >>> upper_triangular = lucid.triu(a, diagonal=1)
    >>> print(upper_triangular)
    Tensor([[0. 2. 3.]
            [0. 0. 6.]
            [0. 0. 0.]], grad=None)

.. note::

    - Only elements in the upper triangular part (including the specified diagonal)  
      are retained; the rest are set to zero.
    - The gradient is passed only through the upper triangular portion,  
      ensuring correct backpropagation.
    - This function is useful for constructing masked matrices in deep learning applications.