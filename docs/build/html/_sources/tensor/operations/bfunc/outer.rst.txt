lucid.outer
===========

.. autofunction:: lucid.outer

The `outer` function computes the outer product of two tensors. 

This operation results in a new tensor formed by multiplying each element 
of the first tensor with each element of the second tensor.

Function Signature
------------------

.. code-block:: python

    def outer(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first input tensor.
- **b** (*Tensor*): The second input tensor.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the outer product result. 
    If either `a` or `b` requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `outer` operation is:

.. math::

    \mathbf{out}_{ij} = \mathbf{a}_i \mathbf{b}_j

For vectors:

.. math::

    \mathbf{out} = 
    \begin{bmatrix} 
    \mathbf{a}_1 \mathbf{b}_1 & \mathbf{a}_1 \mathbf{b}_2 & \dots \\
    \mathbf{a}_2 \mathbf{b}_1 & \mathbf{a}_2 \mathbf{b}_2 & \dots \\
    \vdots & \vdots & \ddots
    \end{bmatrix}

For higher-dimensional tensors, the tensors are first flattened into vectors, and the outer product is calculated.

Backward Gradient Calculation
-----------------------------

For tensors **a** and **b** involved in the `outer` operation, 
the gradients with respect to the output (**out**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \sum_j \mathbf{b}_j

**Gradient with respect to** :math:`\mathbf{b}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{b}} = \sum_i \mathbf{a}_i

For higher-dimensional tensors:

- **Gradient with respect to** :math:`\mathbf{a}`: 
    propagates back along the flattened version of :math:`\mathbf{a}`.
- **Gradient with respect to** :math:`\mathbf{b}`: 
    propagates back along the flattened version of :math:`\mathbf{b}`.

Examples
--------

Using `outer` for a simple outer product:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1.0, 2.0], requires_grad=True)
    >>> b = Tensor([3.0, 4.0], requires_grad=True)
    >>> out = lucid.outer(a, b)
    >>> print(out)
    Tensor([[3.0, 4.0], [6.0, 8.0]], grad=None)

Backpropagation computes gradients for both `a` and `b`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [7.0, 7.0]  # Sum along b
    >>> print(b.grad)
    [3.0, 9.0]  # Sum along a

Using `outer` for higher-dimensional tensors:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    >>> out = lucid.outer(a, b)
    >>> print(out)
    Tensor(
        [[ 5.0,  6.0,  7.0,  8.0],
         [10.0, 12.0, 14.0, 16.0],
         [15.0, 18.0, 21.0, 24.0],
         [20.0, 24.0, 28.0, 32.0]], grad=None)

Backpropagation propagates gradients through the tensors:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [[26.0, 26.0], [26.0, 26.0]]  # Gradients aggregated for a
    >>> print(b.grad)
    [[10.0, 26.0], [18.0, 34.0]]  # Gradients aggregated for b