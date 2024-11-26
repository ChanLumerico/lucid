lucid.min
=========

.. autofunction:: lucid.min

The `min` function computes the minimum value of a tensor along specified axes. 
It is used to perform reduction operations, extracting the smallest elements from the input tensor 
based on the provided axis or axes. This function is useful in various neural network operations, 
such as finding the minimum activation or during certain pooling operations.

Function Signature
------------------

.. code-block:: python

    def min(a: Tensor, axis: int | tuple[int] | None = None, keepdims: bool = False) -> Tensor

Parameters
----------

- **a** (*Tensor*): 
    The input tensor from which to compute the minimum values.

- **axis** (*int* or *tuple of int*, optional): 
    The axis or axes along which to compute the minimum. 
    If `None`, the minimum is computed over all elements 
    of the input tensor. Default is `None`.

- **keepdims** (*bool*, optional): 
    Whether to retain the reduced dimensions with size one. 
    If `True`, the output tensor will have the same number of dimensions as the input tensor, 
    with the reduced dimensions set to size one. Default is `False`.

Returns
-------

- **Tensor**: 
    A new `Tensor` containing the minimum values. The shape of the output tensor 
    depends on the `axis` and `keepdims` parameters. 
    If any of `a` requires gradients, the resulting tensor will also require gradients.

Forward Calculation
-------------------

The forward calculation for the `min` operation is:

.. math::

    \mathbf{out} = \min(\mathbf{a}, \text{axis}= \text{axis}, 
    \text{keepdims}= \text{keepdims})

Where the minimum is computed along the specified axis or axes.

Backward Gradient Calculation
-----------------------------

For the tensor **a** involved in the `min` operation, 
the gradient with respect to the output (**out**) is computed as follows:

**Gradient with respect to** :math:`\mathbf{a}`:

.. math::

    \frac{\partial \mathbf{out}}{\partial \mathbf{a}} = 
    \begin{cases}
        1 & \text{if } \mathbf{a} \text{ is the minimum in its reduction window} \\
        0 & \text{otherwise}
    \end{cases}

This means that the gradient is propagated only to the elements that are 
the minimum in their respective reduction windows.

Examples
--------

Using `min` to find the minimum value in a tensor:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([3.0, 1.0, 2.0], requires_grad=True)
    >>> out = lucid.min(a)
    >>> print(out)
    Tensor(1.0, grad=None)

Backpropagation computes gradients for `a`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [0.0, 1.0, 0.0]

Using `min` along a specific axis with `keepdims`:

.. code-block:: python

    >>> import lucid
    >>> a = Tensor([[3.0, 1.0, 2.0],
    ...            [4.0, 0.5, 6.0]], requires_grad=True)
    >>> out = lucid.min(a, axis=1, keepdims=True)
    >>> print(out)
    Tensor([[1.0],
            [0.5]], grad=None)

Backpropagation computes gradients for `a`:

.. code-block:: python

    >>> out.backward()
    >>> print(a.grad)
    [[0.0, 1.0, 0.0],
     [0.0, 1.0, 0.0]]
