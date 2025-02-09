lucid.einops.reduce
===================

.. autofunction:: lucid.einops.reduce

The `reduce` function performs reduction operations along specified tensor 
dimensions using Einstein notation-like patterns. This operation enables 
summation or averaging over designated axes in a structured manner, 
facilitating various tensor transformations for deep learning applications.

Function Signature
------------------

.. code-block:: python

    def reduce(
        a: Tensor, pattern: _EinopsPattern, reduction: _ReduceStr = "sum", **shapes: int
    ) -> Tensor

Parameters
----------

- **a** (*Tensor*):
  The input tensor to be reduced.

- **pattern** (*_EinopsPattern*):
  A string representing the transformation pattern, specifying dimension 
  reduction using Einstein notation-like syntax.

- **reduction** (*_ReduceStr*, optional):
  The reduction operation to apply, which can be:
  - **"sum"**: Computes the sum along specified dimensions.
  - **"mean"**: Computes the mean along specified dimensions.
  Default is **"sum"**.

- **shapes** (*dict[str, int]*, optional):
  Named dimension sizes that resolve symbolic axes in the pattern.

Returns
-------

- **Tensor**:
  A tensor with reduced dimensions based on the specified pattern and reduction operation.

Mathematical Definition
-----------------------

Given an input tensor :math:`\mathbf{A}` with shape :math:`(d_1, d_2, \dots, d_n)`, 
the `reduce` function applies a transformation defined by the pattern and reduction method:

For **sum reduction**:

.. math::

   \mathbf{B}_{i_1, i_2, \dots} = \sum_{j \in R} 
   \mathbf{A}_{i_1, i_2, \dots, j}

For **mean reduction**:

.. math::

   \mathbf{B}_{i_1, i_2, \dots} = \frac{1}{|R|} \sum_{j \in R} 
   \mathbf{A}_{i_1, i_2, \dots, j}

where :math:`R` represents the reduced indices.

Examples
--------

**Example 1: Summing over one axis**

.. code-block:: python

    >>> import lucid.einops as einops
    >>> a = lucid.Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
    >>> b = einops.reduce(a, "h w -> h", reduction="sum")
    >>> print(b)
    Tensor([3, 7])

**Example 2: Computing mean along an axis**

.. code-block:: python

    >>> a = lucid.Tensor([[1, 2], [3, 4]])  # Shape: (2, 2)
    >>> b = einops.reduce(a, "h w -> h", reduction="mean")
    >>> print(b)
    Tensor([1.5, 3.5])

.. warning::

    When using `mean` reduction, ensure the selected dimensions are 
    meaningful for averaging to avoid unintended numerical scaling effects.

.. important::

    Reduction should preserve the consistency of tensor shapes required for 
    further computations in a neural network.

Advantages
----------

- **Declarative syntax**: Reduces the need for explicit loops.
- **Optimized performance**: Efficient tensor reduction using Einstein notation.
- **Seamless integration**: Works flexibly across different tensor shapes.

Conclusion
----------

The `lucid.einops.reduce` function enables structured reduction operations in `lucid`, 
leveraging Einstein notation for clarity and efficiency in deep learning workflows.

