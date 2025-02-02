lucid.tril
==========

.. autofunction:: lucid.tril

The `tril` function extracts the lower triangular part of a tensor,  
setting all elements above a specified diagonal to zero.

Function Signature
------------------

.. code-block:: python

    def tril(input_: Tensor, diagonal: int = 0) -> Tensor

Parameters
----------

- **input_** (*Tensor*):  
  The input tensor, which must be at least 2-dimensional.  

- **diagonal** (*int, optional*):  
  Specifies which diagonal to consider (default: `0`).  
  - `0`: Keeps the main diagonal.  
  - `>0`: Includes elements above the main diagonal.  
  - `<0`: Excludes more lower rows.

Returns
-------

- **Tensor**:  
  A new tensor with the same shape as the input,  
  but with all elements above the specified diagonal set to zero.

Forward Calculation
-------------------

The `tril` operation applies a lower-triangular mask to the input tensor:

.. math::

    \mathbf{out}_{ij} =
    \begin{cases}
    \mathbf{X}_{ij}, & \text{if } i - j \geq -\text{diagonal} \\
    0, & \text{otherwise}
    \end{cases}

Backward Gradient Calculation
-----------------------------

The gradient for the `tril` operation ensures that only the lower triangular  
portion receives gradients, setting the upper triangular part to zero:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{X}} =
    \mathbf{M}, \quad \text{where} \quad
    M_{ij} =
    \begin{cases}
    1, & \text{if } i - j \geq -\text{diagonal} \\
    0, & \text{otherwise}
    \end{cases}

Example
-------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0, 3.0],
    ...             [4.0, 5.0, 6.0],
    ...             [7.0, 8.0, 9.0]], requires_grad=True)
    >>> lower_triangular = lucid.tril(a)
    >>> print(lower_triangular)
    Tensor([[1. 0. 0.]
            [4. 5. 0.]
            [7. 8. 9.]], grad=None)

    >>> lower_triangular = lucid.tril(a, diagonal=1)
    >>> print(lower_triangular)
    Tensor([[1. 2. 0.]
            [4. 5. 6.]
            [7. 8. 9.]], grad=None)

.. note::

    - Only elements in the lower triangular part (including the specified diagonal)  
      are retained; the rest are set to zero.
    - The gradient is passed only through the lower triangular portion,  
      ensuring correct backpropagation.
    - This function is useful for constructing masked matrices in deep learning applications.