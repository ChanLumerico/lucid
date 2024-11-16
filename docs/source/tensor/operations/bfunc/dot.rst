lucid.dot
=========

.. autofunction:: lucid.dot

The `dot` function computes the dot product of two tensors. 
It is used for vector inner products, matrix-vector products, 
or matrix-matrix multiplications, depending on the shapes of the input tensors.

Function Signature
------------------

.. code-block:: python

    def dot(a: Tensor, b: Tensor) -> Tensor

Parameters
----------

- **a** (*Tensor*): The first input tensor.
- **b** (*Tensor*): The second input tensor.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the dot product result. 
    If either `a` or `b` requires gradients, the resulting tensor 
    will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `dot` operation is:

.. math::

    \mathbf{out} = \mathbf{a} \cdot \mathbf{b}

For vectors:

.. math::

    \mathbf{out} = \sum_{i} \mathbf{a}_i \mathbf{b}_i

For matrices:

.. math::

    \mathbf{out}_{ij} = \sum_{k} \mathbf{a}_{ik} \mathbf{b}_{kj}

Backward Gradient Calculation
-----------------------------

For tensors **a** and **b** involved in the `dot` operation, 
the gradients with respect to the output (**out**) are computed as follows:

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = \mathbf{b}

**Gradient with respect to** :math:`\mathbf{b}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{b}} = \mathbf{a}

For matrix multiplication:

- **Gradient with respect to** :math:`\mathbf{a}`:

  .. math::

      \frac{\partial \mathbf{out}_{ij}}{\partial \mathbf{a}_{ik}} = \mathbf{b}_{kj}

- **Gradient with respect to** :math:`\mathbf{b}`:

  .. math::

      \frac{\partial \mathbf{out}_{ij}}{\partial \mathbf{b}_{kj}} = \mathbf{a}_{ik}

Examples
--------

Using `dot` for a simple dot product:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    >>> b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
    >>> out = lucid.dot(a, b)  # or a.dot(b)
    >>> print(out)
    Tensor(32.0, grad=None)

Backpropagation computes gradients for both `a` and `b`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [4.0, 5.0, 6.0]  # Corresponding to b
    >>> print(b.grad)
    [1.0, 2.0, 3.0]  # Corresponding to a

Using `dot` for matrix multiplication:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    >>> b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    >>> out = lucid.dot(a, b)  # or a.dot(b)
    >>> print(out)
    Tensor([[19.0, 22.0], [43.0, 50.0]], grad=None)

Backpropagation propagates gradients through the matrices:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [[5.0, 7.0], [5.0, 7.0]]
    >>> print(b.grad)
    [[1.0, 3.0], [2.0, 4.0]]
