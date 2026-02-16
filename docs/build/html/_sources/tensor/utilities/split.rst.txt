lucid.split
===========

.. autofunction:: lucid.split

The `split` function divides a tensor into multiple sub-tensors along a specified axis. 
It supports both equal-sized splits and custom-sized splits, making it useful for 
partitioning data in deep learning applications.

Function Signature
------------------

.. code-block:: python

    def split(
        a: Tensor, size_or_sections: int | list[int] | tuple[int], axis: int = 0
    ) -> tuple[Tensor, ...]

Parameters
----------
- **a** (Tensor):
  The input tensor to be split.

- **size_or_sections** (int | list[int] | tuple[int]):
  If an integer, the tensor is split into equal parts along the specified axis.
  If a list or tuple, it specifies the sizes of each split.

- **axis** (int, optional):
  The axis along which to split the tensor. Default is `0`.

Mathematical Expression
------------------------
If `size_or_sections` is an integer `k`, the operation performs:

.. math::

    \text{split}(A, k, \text{axis}) \Rightarrow \{ A_1, A_2, ..., A_k \},

where each sub-tensor satisfies:

.. math::

    A_i \in \mathbb{R}^{(s_1, s_2, ..., s_n)},

and `s_i` are determined based on the specified `axis` and split sizes.

Return Values
-------------
- **tuple[Tensor, ...]**:
  A tuple containing the resulting sub-tensors after the split operation.

Examples
--------
.. code-block:: python

    from lucid import Tensor, split

    x = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    split_tensors = split(x, 2, axis=1)  # Splits into two tensors along axis 1

    print(split_tensors[0].data)  # Output: [[1, 2], [5, 6]]
    print(split_tensors[1].data)  # Output: [[3, 4], [7, 8]]

.. note::

  - If `size_or_sections` is an integer, 
    the input tensor must be evenly divisible along the specified axis.

  - If `size_or_sections` is a list, the sum of its elements must match 
    the size of the tensor along the given axis.
