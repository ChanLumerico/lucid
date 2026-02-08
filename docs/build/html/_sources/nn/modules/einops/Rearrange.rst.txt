nn.Rearrange
============

.. autoclass:: lucid.nn.Rearrange

The `Rearrange` module is a neural network-compatible version of `lucid.einops.rearrange`. 
It provides a structured way to integrate tensor rearrangement into `lucid.nn.Module` 
architectures, ensuring compatibility with model pipelines.

Class Signature
---------------

.. code-block:: python

    class Rearrange(nn.Module):
        def __init__(self, pattern: _EinopsPattern, **shapes: int) -> None

Parameters
----------

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
the `Rearrange` module applies a transformation based on the provided pattern. 
This transformation follows Einstein notation rules:

.. math::

    \mathbf{B}_{i_1, i_2, \dots, i_m} = \sum_{j_1, j_2, \dots, j_n} 
    \mathbf{A}_{j_1, j_2, \dots, j_n} \delta_{(j \rightarrow i)}

where :math:`\delta_{(j \rightarrow i)}` represents a Kronecker delta function 
that enforces index mapping according to the pattern.

The transformation may involve:

- **Permutation**: Swapping dimensions as per pattern constraints.
- **Merging**: Combining multiple dimensions using multiplication.
- **Transposition**: Reordering axes.

Examples
--------

**Integrating `Rearrange` into a neural network**

.. code-block:: python

    >>> import lucid.nn as nn
    >>> model = nn.Sequential(
    ...     nn.Conv2d(3, 16, kernel_size=3),
    ...     nn.Rearrange("b c h w -> b (h w) c")
    ... )
    >>> x = lucid.random.randn(2, 3, 32, 32)  # Shape: (2, 3, 32, 32)
    >>> out = model(x)
    >>> print(out.shape)
    (2, 1024, 16)

.. warning::

    Ensure that any reshaping respects the total number of elements in the tensor.
    Mismatched sizes will result in an error.

.. important::

    The `Rearrange` module follows a declarative approach, 
    meaning you specify **what** the transformation should be, rather than 
    **how** to achieve it computationally.

Advantages
----------

- **Seamless integration**: Works directly within `lucid.nn.Module` models.
- **Declarative syntax**: Einstein notation simplifies tensor operations.
- **Efficient reshaping**: Avoids explicit loops, improving readability and performance.

Conclusion
----------

The `lucid.nn.Rearrange` module brings the power of `einops.rearrange` into `lucid.nn`, 
allowing for flexible tensor transformations within deep learning architectures.
