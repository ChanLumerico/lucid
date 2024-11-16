lucid.inner
===========

.. autofunction:: lucid.inner

The `inner` function computes the inner product of two tensors along the 
last dimension of the first tensor and the second-to-last dimension of 
the second tensor. 

This function generalizes the dot product for tensors of higher dimensions.

Function Signature
------------------

.. code-block:: python

    def inner(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first input tensor.
- **b** (*Tensor*): The second input tensor.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the inner product result. 
    If either `a` or `b` requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `inner` operation is:

.. math::

    \mathbf{out} = \text{sum}(\mathbf{a} \cdot \mathbf{b}, \text{axis}=-1)

For vectors:

.. math::

    \mathbf{out} = \sum_{i} \mathbf{a}_i \mathbf{b}_i

For matrices:

.. math::

    \mathbf{out}_{ij} = \sum_{k} \mathbf{a}_{ik} \mathbf{b}_{jk}

Backward Gradient Calculation
-----------------------------

For tensors **a** and **b** involved in the `inner` operation, 
the gradients with respect to the output (**out**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \mathbf{b}

**Gradient with respect to** :math:`\mathbf{b}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{b}} = \mathbf{a}

For higher-dimensional tensors:

- **Gradient with respect to** :math:`\mathbf{a}`: 
    propagates back along the aligned dimensions of :math:`\mathbf{a}`.
- **Gradient with respect to** :math:`\mathbf{b}`: 
    propagates back along the aligned dimensions of :math:`\mathbf{b}`.

Examples
--------

Using `inner` for a simple inner product:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    >>> out = lucid.inner(a, b)
    >>> print(out)
    Tensor(32.0, grad=None)

Backpropagation computes gradients for both `a` and `b`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [4.0, 5.0, 6.0]  # Corresponding to b
    >>> print(b.grad)
    [1.0, 2.0, 3.0]  # Corresponding to a

Using `inner` for higher-dimensional tensors:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    >>> out = lucid.inner(a, b)
    >>> print(out)
    Tensor([17.0, 53.0], grad=None)

Backpropagation propagates gradients through the tensors:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [[5.0, 6.0], [7.0, 8.0]]
    >>> print(b.grad)
    [[1.0, 2.0], [3.0, 4.0]]
