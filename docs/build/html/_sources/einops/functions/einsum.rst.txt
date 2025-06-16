lucid.einops.einsum
===================

.. autofunction:: lucid.einops.einsum

The `einsum` function performs Einstein summation on one or more input tensors 
according to a specified subscript notation. It enables concise specification 
of complex tensor operations such as contractions, transpositions, reductions, 
and outer products.

Function Signature
------------------

.. code-block:: python

    def einsum(pattern: str, *tensors: Tensor) -> Tensor

Parameters
----------

- **pattern** (*str*):  
  A string in Einstein summation notation (e.g., `'ij,jk->ik'`) specifying the 
  tensor operation.

- **tensors** (*Tensor*):  
  One or more input tensors that match the labels described in `pattern`.

Returns
-------

- **Tensor**:  
  A new tensor resulting from the Einstein summation. Its shape depends on the 
  output subscript in the pattern.

Forward Calculation
-------------------

Einstein summation applies the convention of summing over repeated indices.  
For example:

.. code-block:: python

    einsum('ij,jk->ik', A, B)

is equivalent to:

.. math::

    C_{ik} = \sum_j A_{ij} B_{jk}

This generalizes to multi-input, high-dimensional tensor algebra.

.. tip::
   `einsum` allows flexible and efficient expression of linear algebra, 
   including batched operations.

Examples
--------

**Matrix multiplication:**

.. code-block:: python

    >>> A = Tensor([[1, 2], [3, 4]])
    >>> B = Tensor([[5, 6], [7, 8]])
    >>> C = einsum('ij,jk->ik', A, B)
    >>> print(C)
    Tensor([[19, 22], [43, 50]], grad=None)

**Dot product:**

.. code-block:: python

    >>> a = Tensor([1, 2, 3])
    >>> b = Tensor([4, 5, 6])
    >>> result = einsum('i,i->', a, b)
    >>> print(result)
    Tensor(32, grad=None)

**Batch matrix multiplication:**

.. code-block:: python

    >>> x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> y = Tensor([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
    >>> z = einsum('bij,bjk->bik', x, y)
    >>> print(z)
    Tensor([[[1, 2], [3, 4]], [[11, 11], [15, 15]]], grad=None)

Notes
-----

- Fully supports autodiff in Lucid: gradients are propagated back to all input tensors.
- Ensures GPU/CPU compatibility with device matching.
- Subscript validation errors will raise runtime exceptions.

.. warning::

   Ensure input shapes and labels are consistent with the `pattern`.
