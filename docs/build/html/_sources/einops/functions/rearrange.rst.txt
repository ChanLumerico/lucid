lucid.einops.rearrange
======================

.. autofunction:: lucid.einops.rearrange

The `rearrange` function provides a flexible and expressive way to manipulate 
tensor dimensions using Einstein notation-like patterns. This operation allows 
for dimension permutation, expansion, contraction, and reshaping in a concise 
and intuitive manner.

Function Signature
------------------

.. code-block:: python

    def rearrange(a: Tensor, pattern: _EinopsPattern, **shapes: int) -> Tensor

Parameters
----------

- **a** (*Tensor*):
  The input tensor to be rearranged.

- **pattern** (*_EinopsPattern*):
  A string representing the transformation pattern, 
  using Einstein notation-like syntax to specify dimension permutations and reshaping.

- **shapes** (*dict[str, int]*, optional):
  Named dimension sizes that resolve symbolic axes in the pattern.

Returns
-------

- **Tensor**:
  A tensor with rearranged dimensions according to the specified pattern.

Mathematical Definition
-----------------------

Given an input tensor :math:`\mathbf{A}` with shape :math:`(d_1, d_2, \dots, d_n)`, 
the `rearrange` function applies a transformation based on the provided pattern. 
This transformation can be understood in terms of Einstein notation:

.. math::
    
   \mathbf{B}_{i_1, i_2, \dots, i_m} = \sum_{j_1, j_2, \dots, j_n} 
   \mathbf{A}_{j_1, j_2, \dots, j_n} \delta_{(j \rightarrow i)}

where :math:`\delta_{(j \rightarrow i)}` represents a Kronecker delta function 
enforcing index mapping according to the `pattern`. 

The transformation may involve:

- **Permutation**: Swapping dimensions as per pattern constraints.
- **Merging**: Combining multiple dimensions using multiplication.
- **Transposition**: Reordering axes.

Examples
--------

**Flattening a matrix**

.. code-block:: python

    >>> import lucid.einops as einops
    >>> a = lucid.Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
    >>> b = einops.rearrange(a, "h w -> (h w)")
    >>> print(b)
    Tensor([1, 2, 3, 4])

**Transposing a tensor**

.. code-block:: python

    >>> import lucid.einops as einops
    >>> a = lucid.random.randn(2, 3, 4, 5)  # Shape: (2, 3, 4, 5)
    >>> b = einops.rearrange(a, "n c h w -> n h w c")
    >>> print(b.shape)
    (2, 4, 5, 3)

.. warning::

    Ensure that any reshaping respects the total number of elements in the tensor.
    Mismatched sizes will result in an error.

.. important::

    The `rearrange` function follows a declarative approach, 
    meaning you specify **what** the transformation should be, rather than 
    **how** to achieve it computationally.

Advantages
----------

- **Concise syntax**: Einstein notation simplifies tensor operations.
- **Eliminates explicit loops**: Avoids verbose dimension manipulation.
- **Supports flexible reshaping**: Works seamlessly across various tensor shapes.

Conclusion
----------

The `lucid.einops.rearrange` function enables efficient tensor transformations 
in `lucid`, leveraging a notation inspired by Einstein summation for clarity 
and expressiveness.
