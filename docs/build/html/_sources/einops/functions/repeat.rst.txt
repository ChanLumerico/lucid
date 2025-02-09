lucid.einops.repeat
===================

.. autofunction:: lucid.einops.repeat

The `repeat` function provides a flexible way to duplicate tensor elements along 
specified dimensions using an Einstein notation-like pattern. This operation enables 
controlled expansion and replication of elements to fit a desired output shape.

Function Signature
------------------

.. code-block:: python

    def repeat(a: Tensor, pattern: _EinopsPattern, **shapes: int) -> Tensor

Parameters
----------

- **a** (*Tensor*):
  The input tensor to be repeated.

- **pattern** (*_EinopsPattern*):
  A string representing the repetition pattern, using Einstein notation-like 
  syntax to specify dimension expansion.

- **shapes** (*dict[str, int]*, optional):
  Named dimension sizes that resolve symbolic axes in the pattern.

Returns
-------

- **Tensor**:
  A tensor with repeated elements according to the specified pattern.

Mathematical Definition
-----------------------

Given an input tensor :math:`\mathbf{A}` with shape :math:`(d_1, d_2, \dots, d_n)`, 
the `repeat` function expands selected dimensions by repeating elements. 

The transformation follows the rule:

.. math::

    \mathbf{B}_{i_1, i_2, \dots, i_m} = \mathbf{A}_{j_1, j_2, \dots, j_n}

where indices :math:`i_k` are mapped from :math:`j_k` through the `pattern` by replication.

The transformation may involve:

- **Broadcasting**: Expanding singleton dimensions to specified sizes.
- **Tiling**: Repeating elements along a given dimension.
- **Stacking**: Adding new dimensions by duplicating existing elements.

Examples
--------

**Repeating a vector along a new dimension**

.. code-block:: python

    >>> import lucid.einops as einops
    >>> a = lucid.Tensor([1, 2, 3])  # Shape: (3,)
    >>> b = einops.repeat(a, "i -> i j", j=2)
    >>> print(b)
    Tensor([[1, 1],
            [2, 2],
            [3, 3]])

**Tiling an image tensor**

.. code-block:: python

    >>> import lucid.einops as einops
    >>> a = lucid.random.randn(1, 3, 3)  # Shape: (1, 3, 3)
    >>> b = einops.repeat(a, "b h w -> (b r) h w", r=4)
    >>> print(b.shape)
    (4, 3, 3)

.. warning::

    Ensure that the total number of elements before and after repetition matches.
    Mismatched sizes will result in an error.

.. important::

    The `repeat` function follows a declarative approach, meaning you specify 
    **what** transformation should occur, rather than **how** to compute it explicitly.

Advantages
----------

- **Concise syntax**: Einstein notation simplifies tensor expansions.
- **Efficient tiling**: Eliminates the need for explicit loops.
- **Flexible broadcasting**: Works seamlessly for batch-wise and element-wise expansion.

Conclusion
----------

The `lucid.einops.repeat` function enables controlled repetition of tensor 
elements in `lucid`, leveraging an Einstein summation-inspired notation for 
clarity and expressiveness.
