lucid.matmul
============

.. autofunction:: lucid.matmul

The `matmul` function performs matrix multiplication between two tensors, 
similar to NumPy’s `np.matmul` or the `@` operator.

Function Signature
------------------

.. code-block:: python

    def matmul(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first input tensor.
- **b** (*Tensor*): The second input tensor.

Returns
-------

- **Tensor**: 
    A new `Tensor` resulting from the matrix multiplication of **a** and **b**. 
    If either **a** or **b** requires gradients, the resulting tensor will also require gradients.

Matrix Multiplication Rules
---------------------------

- If both inputs are 2D matrices, standard matrix multiplication is performed.

- If either input is higher-dimensional, the function performs batched matrix multiplication.

- For 1D arrays, the behavior is as follows:
  - **If a is 1D and b is 2D**: The result is a 1D vector computed as \( \mathbf{out} = \mathbf{a} \cdot \mathbf{b} \).
  - **If b is 1D and a is 2D**: The result is also a 1D vector, computed along the last axis of **a**.

.. note::

   The `@` operator can be used as an alternative to `matmul` for matrix multiplication. 
   This makes the syntax cleaner and aligns with NumPy conventions. 
   
   See the **`@` Operator** section below for more details.

Forward Calculation
-------------------

The forward calculation for `matmul` is:

.. math::

    \mathbf{out}_{ij} = \sum_k \mathbf{a}_{ik} \mathbf{b}_{kj}

This formula applies element-wise for batched inputs.

Backward Gradient Calculation
-----------------------------

For tensors **a** and **b** involved in the `matmul` operation, 
the gradients with respect to the output (**out**) are computed as:

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \mathbf{b}^T

**Gradient with respect to** :math:`\mathbf{b}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{b}} = \mathbf{a}^T

For batched inputs, gradients are propagated for each batch independently.

`@` Operator
------------

The `@` operator provides a shorthand for matrix multiplication, 
equivalent to calling `matmul(a, b)`. 

This operator has the same functionality as `matmul`, including batched operations.

Example with `@` operator:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    >>> b = Tensor([[5, 6], [7, 8]], requires_grad=True)
    >>> out = a @ b  # Equivalent to matmul(a, b)
    >>> print(out)
    Tensor([[19, 22], [43, 50]], grad=None)

The `@` operator supports the same functionality as `matmul`, including batched operations:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], requires_grad=True)
    >>> b = Tensor([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], requires_grad=True)
    >>> out = a @ b
    >>> print(out)
    Tensor([[[31, 34], [67, 74]], [[155, 166], [211, 226]]], grad=None)
