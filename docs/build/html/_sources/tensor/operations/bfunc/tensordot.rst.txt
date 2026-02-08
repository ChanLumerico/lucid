lucid.tensordot
===============

.. autofunction:: lucid.tensordot

The `tensordot` function performs generalized tensor contraction over specified axes,
extending the concepts of dot product and matrix multiplication to higher-dimensional tensors.

Function Signature
------------------

.. code-block:: python

    def tensordot(
        a: Tensor,
        b: Tensor,
        axes: int | tuple[int, int] | tuple[list[int], list[int]] = 2,
    ) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first input tensor.
- **b** (*Tensor*): The second input tensor.
- **axes** (*int | tuple[int, int] | tuple[list[int], list[int]]*):  
  The axes over which to contract:
  
  - If **int**, it contracts over the last `axes` dimensions of **a** and the first `axes` dimensions of **b**.
  - If **tuple of two ints**, it contracts over the specified dimensions of **a** and **b**.
  - If **tuple of two lists**, the lists explicitly specify multiple axes from **a** and **b** to contract.

Returns
-------

- **Tensor**:  
  The result of the tensor contraction. The shape depends on the uncontracted axes of both inputs.
  Gradient is propagated if either input requires it.

Tensor Contraction Rules
-------------------------

The contraction sums over the specified axes:

**Case 1**: Integer `axes = k`

.. math::

   \text{out}[i_0, \dots, i_{m-k-1}, j_k, \dots, j_{n-1}] =
   \sum_{s_1, \dots, s_k}
   a[i_0, \dots, i_{m-k-1}, s_1, \dots, s_k] \cdot
   b[s_1, \dots, s_k, j_k, \dots, j_{n-1}]

**Case 2**: Tuple of axes `(axes_a, axes_b)`

.. math::

   \text{out} = \sum_{\text{axes}_a = \text{axes}_b}
   a[\dots, i_k, \dots] \cdot b[\dots, i_k, \dots]

The output shape is determined by concatenating the non-contracted axes of **a** and **b**.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]])
    >>> b = Tensor([[5, 6], [7, 8]])
    >>> out = lucid.tensordot(a, b, axes=1)
    >>> print(out)
    Tensor([19, 43], grad=None)

Contract over multiple axes:

.. code-block:: python

    >>> a = Tensor(np.arange(60).reshape(3, 4, 5))
    >>> b = Tensor(np.arange(24).reshape(4, 3, 2))
    >>> out = lucid.tensordot(a, b, axes=([1, 0], [0, 1]))
    >>> out.shape
    (5, 2)

Forward Calculation
-------------------

If contracting over dimensions :math:`i_1, \dots, i_k`, the output is:

.. math::

    \text{out} = \sum_{i_1, \dots, i_k} a[\dots, i_1, \dots, i_k] \cdot b[i_1, \dots, i_k, \dots]

Backward Gradient Calculation
-----------------------------

Let:

.. math::

    \text{out} = \text{tensordot}(a, b, \text{axes})

Then the gradients are:

.. math::

    \frac{\partial \text{out}}{\partial a} = \text{tensordot}(\text{grad}, b^\top, \dots)

.. math::

    \frac{\partial \text{out}}{\partial b} = \text{tensordot}(a^\top, \text{grad}, \dots)

Axis permutation and reshape are handled internally to match broadcasting and contraction.

.. note::

   `tensordot` is more flexible than `matmul` and is useful for computing attention scores, batched inner products, or high-order tensor reductions.
